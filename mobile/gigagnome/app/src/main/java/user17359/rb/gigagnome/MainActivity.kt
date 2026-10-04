package user17359.rb.gigagnome

import android.Manifest
import android.content.Context
import android.content.pm.PackageManager
import android.os.Build
import android.os.Bundle
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.compose.rememberLauncherForActivityResult
import androidx.activity.enableEdgeToEdge
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.OptIn
import androidx.annotation.RequiresApi
import androidx.camera.core.CameraSelector
import androidx.camera.core.ExperimentalGetImage
import androidx.camera.core.ImageAnalysis
import androidx.camera.core.ImageProxy
import androidx.camera.core.Preview
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.camera.view.PreviewView
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.collectAsState
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.saveable.rememberSaveable
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.layout.onGloballyPositioned
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import androidx.compose.ui.viewinterop.AndroidView
import androidx.core.content.ContextCompat
import androidx.lifecycle.compose.LocalLifecycleOwner
import com.google.mlkit.vision.common.InputImage
import com.google.mlkit.vision.pose.PoseDetection
import com.google.mlkit.vision.pose.PoseDetector
import com.google.mlkit.vision.pose.PoseLandmark
import com.google.mlkit.vision.pose.defaults.PoseDetectorOptions
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.asStateFlow
import okhttp3.MediaType.Companion.toMediaType
import okhttp3.RequestBody.Companion.toRequestBody
import user17359.rb.gigagnome.ui.theme.GigagnomeTheme
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

// ============================================================
// Data model
// ============================================================

data class PoseState(
    val landmarks: List<PoseLandmark> = emptyList(),
    val imageWidth: Int = 0,
    val imageHeight: Int = 0,
    // --- debug info shown in the on-screen HUD ---
    val framesAnalyzed: Int = 0,
    val lastError: String? = null,
    val lastImageFormat: Int = 0,
    val lastRotation: Int = 0,
    val bindError: String? = null
)

object PoseHolder {
    private val _pose = MutableStateFlow(PoseState())
    val pose = _pose.asStateFlow()
    fun update(p: PoseState) { _pose.value = p }
    fun updateDebug(error: String?, frames: Int, format: Int, rotation: Int) {
        _pose.value = _pose.value.copy(
            lastError = error,
            framesAnalyzed = frames,
            lastImageFormat = format,
            lastRotation = rotation
        )
    }
    fun updateBindError(error: String?) {
        _pose.value = _pose.value.copy(bindError = error)
    }
}

// ============================================================
// Server configuration (entered on the setup screen)
// ============================================================

object ServerConfig {
    private const val PREFS = "server_config"
    private const val KEY_IP = "server_ip"
    private const val KEY_PORT = "server_port"
    const val DEFAULT_PORT = 5000

    @Volatile var baseUrl: String = ""   // e.g. "http://10.250.194.159:5000"

    fun load(context: Context) {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        val ip = prefs.getString(KEY_IP, null)
        val port = prefs.getInt(KEY_PORT, DEFAULT_PORT)
        if (!ip.isNullOrBlank()) baseUrl = buildUrl(ip, port)
    }

    fun save(context: Context, ip: String, port: Int) {
        baseUrl = buildUrl(ip, port)
        context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
            .edit()
            .putString(KEY_IP, ip)
            .putInt(KEY_PORT, port)
            .apply()
    }

    fun saved(context: Context): Pair<String, Int> {
        val prefs = context.getSharedPreferences(PREFS, Context.MODE_PRIVATE)
        return (prefs.getString(KEY_IP, "") ?: "") to prefs.getInt(KEY_PORT, DEFAULT_PORT)
    }

    private fun buildUrl(ip: String, port: Int) = "http://$ip:$port"

    /** True when a server address has been configured (and loaded). */
    val isConfigured: Boolean get() = baseUrl.isNotBlank()
}

/**
 * Forwards pose data to the server.
 * All analysis (plank quality, timer, feedback) happens server-side.
 */
object PoseSender {
    @RequiresApi(Build.VERSION_CODES.O)
    private val client = okhttp3.OkHttpClient.Builder()
        .connectTimeout(java.time.Duration.ofSeconds(2))
        .build()
    private val json = okhttp3.MediaType
        .run { "application/json; charset=utf-8".toMediaType() }
    // Unique per installed app instance so the server can tell
    // concurrent users apart:
    private val DEVICE_ID: String =
        "dev-" + java.util.UUID.randomUUID().toString().substring(0, 8)
    private const val MIN_INTERVAL_MS = 200L // ~5 frames/sec
    @Volatile private var lastSentAt = 0L

    @RequiresApi(Build.VERSION_CODES.O)
    fun enqueue(landmarks: List<PoseLandmark>, imageWidth: Int, imageHeight: Int) {
        if (!ServerConfig.isConfigured) return // no server configured yet
        val now = android.os.SystemClock.elapsedRealtime()
        if (now - lastSentAt < MIN_INTERVAL_MS) return // throttle
        lastSentAt = now

        val body = org.json.JSONObject().apply {
            put("device_id", DEVICE_ID)
            put("w", imageWidth)
            put("h", imageHeight)
            put("points", org.json.JSONArray().apply {
                for (l in landmarks) {
                    put(org.json.JSONObject().apply {
                        put("type", l.landmarkType)
                        put("x", l.position.x.toDouble())
                        put("y", l.position.y.toDouble())
                        put("likelihood", l.inFrameLikelihood.toDouble())
                    })
                }
            })
        }

        val request = okhttp3.Request.Builder()
            .url("${ServerConfig.baseUrl}/pose")
            .post(body.toString().toRequestBody(json))
            .build()

        // Fire and forget on a background thread; failures are ignored
        // (phone may leave Wi-Fi). Uses blocking execute() instead of
        // Callback, which avoids OkHttp version/signature mismatches.
        Thread {
            try {
                client.newCall(request).execute().use { /* body not needed */ }
            } catch (_: java.io.IOException) {
                // ignored
            }
        }.start()
    }
}

// ============================================================
// Pose analyzer (ImageAnalysis -> ML Kit Pose Detection)
// ============================================================

class PoseAnalyzer : ImageAnalysis.Analyzer {

    private val options = PoseDetectorOptions.Builder()
        .setDetectorMode(PoseDetectorOptions.STREAM_MODE)
        .build()
    private val detector: PoseDetector = PoseDetection.getClient(options)

    @Volatile private var latestTimestamp = 0L
    @Volatile private var frameCount = 0
    @Volatile private var lastError: String? = null

    @RequiresApi(Build.VERSION_CODES.O)
    @OptIn(ExperimentalGetImage::class)
    override fun analyze(image: ImageProxy) {
        val mediaImage = image.image
        if (mediaImage == null || image.imageInfo.timestamp <= latestTimestamp) {
            image.close()
            return
        }
        latestTimestamp = image.imageInfo.timestamp
        frameCount++

        val rotation = image.imageInfo.rotationDegrees
        val input = InputImage.fromMediaImage(mediaImage, rotation)
        // ML Kit returns coordinates in the upright frame, so swap buffer
        // dimensions when the image is rotated 90/270 degrees.
        val width: Int
        val height: Int
        if (rotation == 90 || rotation == 270) {
            width = image.height
            height = image.width
        } else {
            width = image.width
            height = image.height
        }
        val imageFormat = image.format
        // Publish frame info immediately — if frames: stays 0 now,
        // the analyzer itself is not receiving frames from CameraX.
        // If frames: counts up but landmarks stay 0, ML Kit is the problem.
        PoseHolder.updateDebug(lastError, frameCount, imageFormat, rotation)
        // Do NOT close the ImageProxy here — ML Kit is still reading the
        // underlying buffer. Closing early frees the image and the detector
        // silently fails. Close it only when processing completes.

        detector.process(input)
            .addOnSuccessListener { pose ->
                val relevant = listOf(
                    PoseLandmark.NOSE,
                    PoseLandmark.LEFT_SHOULDER, PoseLandmark.RIGHT_SHOULDER,
                    PoseLandmark.LEFT_ELBOW, PoseLandmark.RIGHT_ELBOW,
                    PoseLandmark.LEFT_WRIST, PoseLandmark.RIGHT_WRIST,
                    PoseLandmark.LEFT_HIP, PoseLandmark.RIGHT_HIP,
                    PoseLandmark.LEFT_KNEE, PoseLandmark.RIGHT_KNEE,
                    PoseLandmark.LEFT_ANKLE, PoseLandmark.RIGHT_ANKLE
                ).mapNotNull { type -> pose.getPoseLandmark(type) }

                lastError = null
                PoseHolder.update(
                    PoseState(
                        landmarks = relevant,
                        imageWidth = width,
                        imageHeight = height,
                        framesAnalyzed = frameCount,
                        lastError = lastError,
                        lastImageFormat = imageFormat,
                        lastRotation = rotation
                    )
                )

                // Forward raw pose data to the server; the server will do
                // all calculations (plank quality, timer, etc.).
                PoseSender.enqueue(relevant, width, height)
            }
            .addOnFailureListener {
                lastError = "${it.javaClass.simpleName}: ${it.message}"
                PoseHolder.updateDebug(lastError, frameCount, imageFormat, rotation)
            }
            .addOnCompleteListener {
                // The ONLY place the ImageProxy is closed.
                image.close()
            }
    }
}

// ============================================================
// Server setup screen (shown before recording starts)
// ============================================================

private fun isValidIp(ip: String): Boolean {
    val part = "(25[0-5]|2[0-4]\\d|1\\d\\d|[1-9]?\\d)"
    return Regex("^$part\\.$part\\.$part\\.$part$").matches(ip.trim())
}

@Composable
fun ServerSetupScreen(
    onStart: (ip: String, port: Int) -> Unit
) {
    val context = LocalContext.current
    val (savedIp, savedPort) = ServerConfig.saved(context)

    var ip by rememberSaveable { mutableStateOf(savedIp) }
    var port by rememberSaveable { mutableStateOf(savedPort.toString()) }
    var showError by rememberSaveable { mutableStateOf(false) }

    Column(
        modifier = Modifier
            .fillMaxSize()
            .background(MaterialTheme.colorScheme.background)
            .padding(32.dp),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = androidx.compose.foundation.layout.Arrangement.Center
    ) {
        Text("Plank Tracker", fontSize = 28.sp, color = MaterialTheme.colorScheme.onBackground)
        Spacer(Modifier.height(8.dp))
        Text(
            "Enter the address of the pose server",
            fontSize = 16.sp,
            color = MaterialTheme.colorScheme.onBackground.copy(alpha = 0.7f)
        )
        Spacer(Modifier.height(32.dp))

        OutlinedTextField(
            value = ip,
            onValueChange = { ip = it; showError = false },
            label = { Text("Server IP address") },
            placeholder = { Text("e.g. 10.250.194.159") },
            isError = showError && !isValidIp(ip),
            singleLine = true,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            modifier = Modifier.fillMaxWidth()
        )
        Spacer(Modifier.height(16.dp))
        OutlinedTextField(
            value = port,
            onValueChange = { port = it.filter { c -> c.isDigit() }; showError = false },
            label = { Text("Port") },
            singleLine = true,
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            modifier = Modifier.fillMaxWidth()
        )

        if (showError) {
            Spacer(Modifier.height(8.dp))
            Text(
                "Please enter a valid IP address (e.g. 192.168.1.10)",
                color = MaterialTheme.colorScheme.error,
                fontSize = 14.sp
            )
        }

        Spacer(Modifier.height(32.dp))
        Button(
            onClick = {
                val p = port.toIntOrNull() ?: ServerConfig.DEFAULT_PORT
                if (isValidIp(ip)) {
                    ServerConfig.save(context, ip.trim(), p)
                    onStart(ip.trim(), p)
                } else {
                    showError = true
                }
            },
            modifier = Modifier.fillMaxWidth()
        ) {
            Text("Start recording", fontSize = 18.sp)
        }
    }
}

// ============================================================
// Camera composable
// ============================================================

@Composable
fun PlankTrackerScreen(
    onBackToSetup: () -> Unit = {},
    modifier: Modifier = Modifier
) {
    val context = LocalContext.current

    var hasPermission by remember {
        mutableStateOf(
            ContextCompat.checkSelfPermission(context, Manifest.permission.CAMERA)
                    == PackageManager.PERMISSION_GRANTED
        )
    }
    val permissionLauncher = rememberLauncherForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { granted -> hasPermission = granted }

    LaunchedEffect(Unit) {
        if (!hasPermission) permissionLauncher.launch(Manifest.permission.CAMERA)
    }

    Box(modifier = modifier.fillMaxSize().background(Color.Black)) {
        if (hasPermission) {
            CameraWithOverlay(onBackToSetup = onBackToSetup)
        } else {
            Column(
                modifier = Modifier.align(Alignment.Center).padding(32.dp),
                horizontalAlignment = Alignment.CenterHorizontally
            ) {
                Text("Camera permission needed", color = Color.White, fontSize = 18.sp)
                Spacer(Modifier.height(12.dp))
                Button(onClick = { permissionLauncher.launch(Manifest.permission.CAMERA) }) {
                    Text("Grant permission")
                }
            }
        }
    }
}

@Composable
fun CameraWithOverlay(onBackToSetup: () -> Unit = {}) {
    val context = LocalContext.current
    val lifecycleOwner = LocalLifecycleOwner.current
    val poseState by PoseHolder.pose.collectAsState()

    var previewViewRef by remember { mutableStateOf<PreviewView?>(null) }
    var viewWidth by remember { mutableStateOf(0) }
    var viewHeight by remember { mutableStateOf(0) }
    val analysisExecutor: ExecutorService = remember { Executors.newSingleThreadExecutor() }

    DisposableEffect(Unit) {
        var cameraProvider: ProcessCameraProvider? = null
        val future = ProcessCameraProvider.getInstance(context)
        future.addListener({
            try {
                val provider = future.get()
                cameraProvider = provider

                val preview = Preview.Builder().build().also {
                    it.setSurfaceProvider(previewViewRef?.surfaceProvider)
                }
                val analysis = ImageAnalysis.Builder()
                    .setBackpressureStrategy(ImageAnalysis.STRATEGY_KEEP_ONLY_LATEST)
                    .build()
                    .also { it.setAnalyzer(analysisExecutor, PoseAnalyzer()) }

                val selector = CameraSelector.Builder()
                    .requireLensFacing(CameraSelector.LENS_FACING_FRONT)
                    .build()

                provider.unbindAll()
                provider.bindToLifecycle(lifecycleOwner, selector, preview, analysis)
                PoseHolder.updateBindError(null)
            } catch (e: Exception) {
                // Surface the failure on the HUD instead of swallowing it.
                PoseHolder.updateBindError("${e.javaClass.simpleName}: ${e.message}")
            }
        }, ContextCompat.getMainExecutor(context))

        onDispose {
            analysisExecutor.shutdown()
            runCatching { cameraProvider?.unbindAll() }
        }
    }

    Box(Modifier.fillMaxSize()) {
        AndroidView(
            factory = { ctx -> PreviewView(ctx).apply { previewViewRef = this } },
            modifier = Modifier.fillMaxSize()
        )
        // Overlay that also tracks view size for coordinate mapping
        Box(
            Modifier
                .fillMaxSize()
                .onGloballyPositioned { coords ->
                    viewWidth = coords.size.width
                    viewHeight = coords.size.height
                }
        ) {
            PoseOverlay(
                poseState = poseState,
                viewWidth = viewWidth,
                viewHeight = viewHeight,
                mirrored = true, // front camera preview is mirrored
                modifier = Modifier.fillMaxSize()
            )
        }

        // Debug HUD: on-screen diagnostics (no Logcat needed in the field)
        val debugText = buildString {
            append("landmarks: ${poseState.landmarks.size}")
            append("  ${poseState.imageWidth}x${poseState.imageHeight}")
            append("\nframes: ${poseState.framesAnalyzed}")
            append("  rot: ${poseState.lastRotation}")
            append("  fmt: ${poseState.lastImageFormat}")
            poseState.lastError?.let { append("\nERR: $it") }
            poseState.bindError?.let { append("\nBIND: $it") }
        }
        Text(
            text = debugText,
            color = if (poseState.lastError != null) Color.Red else Color.Black,
            fontSize = 20.sp,
            modifier = Modifier
                .align(Alignment.TopStart)
                .padding(8.dp)
        )

        // Small button to go back to the server setup screen
        Button(
            onClick = onBackToSetup,
            modifier = Modifier
                .align(Alignment.TopEnd)
                .padding(8.dp)
        ) {
            Text("Change server")
        }
    }
}

// ============================================================
// Overlay drawing
// ============================================================

private val CONNECTIONS = listOf(
    PoseLandmark.LEFT_SHOULDER to PoseLandmark.RIGHT_SHOULDER,
    PoseLandmark.LEFT_SHOULDER to PoseLandmark.LEFT_ELBOW,
    PoseLandmark.LEFT_ELBOW to PoseLandmark.LEFT_WRIST,
    PoseLandmark.RIGHT_SHOULDER to PoseLandmark.RIGHT_ELBOW,
    PoseLandmark.RIGHT_ELBOW to PoseLandmark.RIGHT_WRIST,
    PoseLandmark.LEFT_SHOULDER to PoseLandmark.LEFT_HIP,
    PoseLandmark.RIGHT_SHOULDER to PoseLandmark.RIGHT_HIP,
    PoseLandmark.LEFT_HIP to PoseLandmark.RIGHT_HIP,
    PoseLandmark.LEFT_HIP to PoseLandmark.LEFT_KNEE,
    PoseLandmark.LEFT_KNEE to PoseLandmark.LEFT_ANKLE,
    PoseLandmark.RIGHT_HIP to PoseLandmark.RIGHT_KNEE,
    PoseLandmark.RIGHT_KNEE to PoseLandmark.RIGHT_ANKLE
)

@Composable
fun PoseOverlay(
    poseState: PoseState,
    viewWidth: Int,
    viewHeight: Int,
    mirrored: Boolean,
    modifier: Modifier = Modifier
) {
    val pointColor = Color(0xFF4CAF50)
    val animatedAlpha by animateFloatAsState(
        if (poseState.landmarks.isNotEmpty()) 1f else 0f
    )

    Canvas(modifier = modifier) {
        if (poseState.landmarks.isEmpty() ||
            poseState.imageWidth == 0 || poseState.imageHeight == 0 ||
            viewWidth <= 1 || viewHeight <= 1
        ) return@Canvas

        val vw = size.width
        val vh = size.height

        // Map image coords -> view coords assuming the preview fills the box
        // (PreviewView default scaleType = FILL_CENTER).
        val imageAspect = poseState.imageWidth.toFloat() / poseState.imageHeight
        val viewAspect = vw / vh
        val scale: Float
        var dx = 0f
        var dy = 0f
        if (viewAspect > imageAspect) {
            scale = vw / poseState.imageWidth
            dy = (vh - poseState.imageHeight * scale) / 2f
        } else {
            scale = vh / poseState.imageHeight
            dx = (vw - poseState.imageWidth * scale) / 2f
        }

        val mirror = mirrored // capture for the local map function
        fun map(l: PoseLandmark): Offset {
            var x = l.position.x * scale + dx
            val y = l.position.y * scale + dy
            if (mirror) x = vw - x
            return Offset(x, y)
        }

        val byType = poseState.landmarks.associateBy { it.landmarkType }

        // Skeleton lines
        for ((a, b) in CONNECTIONS) {
            val la = byType[a]; val lb = byType[b]
            if (la != null && lb != null &&
                la.inFrameLikelihood > 0.5f && lb.inFrameLikelihood > 0.5f
            ) {
                drawLine(
                    color = pointColor.copy(alpha = 0.9f * animatedAlpha),
                    start = map(la),
                    end = map(lb),
                    strokeWidth = 6f
                )
            }
        }

        // Keypoints: white halo + colored dot on every detected landmark
        for (l in poseState.landmarks) {
            if (l.inFrameLikelihood > 0.5f) {
                drawCircle(
                    color = Color.White.copy(alpha = animatedAlpha),
                    radius = 10f,
                    center = map(l)
                )
                drawCircle(
                    color = pointColor.copy(alpha = animatedAlpha),
                    radius = 6f,
                    center = map(l)
                )
            }
        }
    }
}

// ============================================================
// Activity
// ============================================================

class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        // Load previously saved server address (if any)
        ServerConfig.load(this)

        setContent {
            GigagnomeTheme {
                Surface(
                    modifier = Modifier.fillMaxSize(),
                    color = MaterialTheme.colorScheme.background
                ) {
                    var serverReady by rememberSaveable { mutableStateOf(ServerConfig.isConfigured) }

                    if (!serverReady) {
                        ServerSetupScreen(
                            onStart = { _, _ -> serverReady = true }
                        )
                    } else {
                        PlankTrackerScreen(
                            onBackToSetup = { serverReady = false }
                        )
                    }
                }
            }
        }
    }
}