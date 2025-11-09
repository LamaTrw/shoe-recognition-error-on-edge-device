package com.example.yololitertobjectdetection

import android.content.Context
import android.graphics.Bitmap
import android.os.SystemClock
import com.example.yololitertobjectdetection.MetaData.extractNamesFromLabelFile
import com.example.yololitertobjectdetection.MetaData.extractNamesFromMetadata
import org.tensorflow.lite.DataType
import org.tensorflow.lite.Interpreter
import org.tensorflow.lite.gpu.CompatibilityList
import org.tensorflow.lite.gpu.GpuDelegate
import org.tensorflow.lite.support.common.FileUtil
import org.tensorflow.lite.support.common.ops.CastOp
import org.tensorflow.lite.support.common.ops.NormalizeOp
import org.tensorflow.lite.support.image.ImageProcessor
import org.tensorflow.lite.support.image.TensorImage
import org.tensorflow.lite.support.tensorbuffer.TensorBuffer
import kotlin.math.max
import kotlin.math.min

class Detector(
    private val context: Context,
    private val modelPath: String,
    private val labelPath: String?,
    private val detectorListener: DetectorListener,
    private val message: (String) -> Unit
) {
    private var interpreter: Interpreter
    private var labels = mutableListOf<String>()

    private var tensorWidth = 0
    private var tensorHeight = 0
    private var numChannel = 0
    private var numElements = 0

    private val imageProcessor = ImageProcessor.Builder()
        .add(NormalizeOp(0f, 255f))
        .add(CastOp(DataType.FLOAT32))
        .build()

    init {
        // Load model
        val options = Interpreter.Options().apply { setNumThreads(4) }
        val model = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(model, options)

        // Load labels
        labels.addAll(extractNamesFromMetadata(model))
        if (labels.isEmpty()) {
            if (labelPath == null) {
                message("Model không có metadata, dùng LABELS_PATH trong Constants.kt")
                labels.addAll(MetaData.TEMP_CLASSES)
            } else {
                labels.addAll(extractNamesFromLabelFile(context, labelPath))
            }
        }

        // Lấy kích thước tensor
        val inputShape = interpreter.getInputTensor(0).shape()
        val outputShape = interpreter.getOutputTensor(0).shape()

        tensorWidth = inputShape[1] // [1, 640, 640, 3] -> 640
        tensorHeight = inputShape[2] // -> 640

        // Output shape [1, 6, 8400]
        numChannel = outputShape[1] // -> 6 (x,y,w,h, cls_0, cls_1)
        numElements = outputShape[2] // -> 8400
    }

    fun restart(isGpu: Boolean) {
        interpreter.close()

        val options = Interpreter.Options().apply {
            if (isGpu) {
                val compatList = CompatibilityList()
                if (compatList.isDelegateSupportedOnThisDevice) {
                    val delegateOptions = compatList.bestOptionsForThisDevice
                    addDelegate(GpuDelegate(delegateOptions))
                } else {
                    setNumThreads(4)
                }
            } else {
                setNumThreads(4)
            }
        }

        val model = FileUtil.loadMappedFile(context, modelPath)
        interpreter = Interpreter(model, options)
    }

    fun close() = interpreter.close()

    fun detect(frame: Bitmap) {
        if (tensorWidth == 0 || tensorHeight == 0 || numChannel == 0 || numElements == 0) return

        var inferenceTime = SystemClock.uptimeMillis()

        // Resize ảnh đúng kích thước model
        val resizedBitmap = Bitmap.createScaledBitmap(frame, tensorWidth, tensorHeight, false)
        val tensorImage = TensorImage(DataType.FLOAT32)
        tensorImage.load(resizedBitmap)
        val processedImage = imageProcessor.process(tensorImage)
        val imageBuffer = processedImage.buffer

        // ✅ SỬA LỖI 1: FLOAT3T -> FLOAT32
        val output = TensorBuffer.createFixedSize(intArrayOf(1, numChannel, numElements), DataType.FLOAT32)
        interpreter.run(imageBuffer, output.buffer)

        // BƯỚC 1: Lọc confidence và class
        val bestBoxes = preFilterBoxes(output.floatArray)
        // BƯỚC 2: Chạy Non-Max Suppression
        val finalBoxes = nonMaxSuppression(bestBoxes)

        inferenceTime = SystemClock.uptimeMillis() - inferenceTime

        if (finalBoxes.isEmpty()) {
            detectorListener.onEmptyDetect()
        } else {
            // Cập nhật FPS nếu cần
            // val fps = 1000f / inferenceTime
            // ...
            detectorListener.onDetect(finalBoxes, inferenceTime)
        }
    }

    private fun preFilterBoxes(array: FloatArray): List<BoundingBox> {
        val boxes = mutableListOf<BoundingBox>()

        // Output format: [1, 6, 8400]
        // 0: x
        // 1: y
        // 2: w
        // 3: h
        // 4: class_0_conf ('stiching')
        // 5: class_1_conf ('bonding')

        for (i in 0 until numElements) { // 8400
            val confClass0 = array[4 * numElements + i]
            val confClass1 = array[5 * numElements + i]

            var maxConfidence: Float
            var maxClassIndex: Int

            // Tìm xem class nào có confidence cao hơn
            if (confClass0 > confClass1) {
                maxConfidence = confClass0
                maxClassIndex = 0
            } else {
                maxConfidence = confClass1
                maxClassIndex = 1
            }

            // Chỉ lấy box nếu confidence cao nhất vượt ngưỡng
            if (maxConfidence > CONFIDENCE_THRESHOLD) {
                val x = array[i]
                val y = array[numElements + i]
                val w = array[2 * numElements + i]
                val h = array[3 * numElements + i]

                val clsName = labels.getOrElse(maxClassIndex) { "unknown" }

                boxes.add(
                    BoundingBox(
                        x1 = x - w / 2, // cx -> x1
                        y1 = y - h / 2, // cy -> y1
                        x2 = x + w / 2, // cx + w -> x2
                        y2 = y + h / 2, // cy + h -> y2
                        cnf = maxConfidence,
                        cls = maxClassIndex,
                        clsName = clsName
                    )
                )
            }
        }

        return boxes
    }

    private fun nonMaxSuppression(boxes: List<BoundingBox>): List<BoundingBox> {
        if (boxes.isEmpty()) return emptyList()

        // ✅ SỬA LỖI 2: val -> var
        var sortedBoxes = boxes.sortedByDescending { it.cnf }
        val finalBoxes = mutableListOf<BoundingBox>()

        while (sortedBoxes.isNotEmpty()) {
            val currentBox = sortedBoxes.first()
            finalBoxes.add(currentBox)

            // Lấy các box còn lại
            val remainingBoxes = sortedBoxes.drop(1).toMutableList()

            val iterator = remainingBoxes.iterator()
            while (iterator.hasNext()) {
                val nextBox = iterator.next()
                // Chỉ so sánh NMS nếu 2 box cùng class
                if (nextBox.cls == currentBox.cls) {
                    if (iou(currentBox, nextBox) > NMS_THRESHOLD) {
                        iterator.remove() // Xóa box nếu nó trùng lặp
                    }
                }
            }
            // Cập nhật list cho vòng lặp tiếp theo
            if (remainingBoxes.size == sortedBoxes.size) break // Tránh lặp vô hạn nếu không xóa gì
            sortedBoxes = remainingBoxes
        }

        return finalBoxes
    }

    private fun iou(box1: BoundingBox, box2: BoundingBox): Float {
        val x1 = max(box1.x1, box2.x1)
        val y1 = max(box1.y1, box2.y1)
        val x2 = min(box1.x2, box2.x2)
        val y2 = min(box1.y2, box2.y2)

        val intersection = max(0f, x2 - x1) * max(0f, y2 - y1)

        val box1Area = (box1.x2 - box1.x1) * (box1.y2 - box1.y1)
        val box2Area = (box2.x2 - box2.x1) * (box2.y2 - box2.y1)

        val union = box1Area + box2Area - intersection

        return if (union == 0f) 0f else intersection / union
    }


    interface DetectorListener {
        fun onEmptyDetect()
        fun onDetect(boundingBoxes: List<BoundingBox>, inferenceTime: Long)
    }

    companion object {
        private const val CONFIDENCE_THRESHOLD = 0.3F
        private const val NMS_THRESHOLD = 0.5F // Ngưỡng NMS
    }
}