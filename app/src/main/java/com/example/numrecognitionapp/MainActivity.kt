package com.example.numrecognitionapp

import android.content.Context
import android.graphics.*
import android.os.Bundle
import android.util.Log
import android.view.View
import android.view.ViewTreeObserver
import androidx.appcompat.app.AppCompatActivity
import kotlinx.android.synthetic.main.activity_main.*
import org.pytorch.IValue
import org.pytorch.Module
import org.pytorch.torchvision.TensorImageUtils
import java.io.File
import java.io.FileOutputStream


class MainActivity : AppCompatActivity() {

    var surfaceViewWidth: Int? = null
    var surfaceViewHeight: Int? = null
    var drawSurfaceView:DrawSurfaceView? = null

    /// 拡張関数
    // ViewTreeObserverを使ってViewが作成されてからsurfaceViewのサイズ取得
    private inline fun <T : View> T.afterMeasure(crossinline f: T.() -> Unit) {
        viewTreeObserver.addOnGlobalLayoutListener(object :
            ViewTreeObserver.OnGlobalLayoutListener {
            override fun onGlobalLayout() {
                if (width > 0 && height > 0) {
                    viewTreeObserver.removeOnGlobalLayoutListener(this)
                    f()
                }
            }
        })
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        setContentView(R.layout.activity_main)


        /// ViewTreeObserberを使用
        /// surfaceViewが生成し終わってからsurfaceViewのサイズを取得
        surfaceView.afterMeasure {
            surfaceViewWidth = surfaceView.width
            surfaceViewHeight = surfaceView.height
            //// DrawrSurfaceViewのセットとインスタンス生成
            drawSurfaceView = DrawSurfaceView(
                applicationContext,
                surfaceView,
                surfaceViewWidth!!,
                surfaceViewHeight!!
            )
            surfaceView.setOnTouchListener { v, event -> drawSurfaceView!!.onTouch(event) }
        }


        /// リセットボタン
        resetBtn.setOnClickListener {
            drawSurfaceView!!.reset()
            sampleImg.setImageResource(R.color.colorPrimaryDark)
            resultNum.text = "？"
        }

        //// assetファイルからパスを取得する関数
        fun assetFilePath(context: Context, assetName: String): String {
            val file = File(context.filesDir, assetName)
            if (file.exists() && file.length() > 0) {
                return file.absolutePath
            }
            context.assets.open(assetName).use { inputStream ->
                FileOutputStream(file).use { outputStream ->
                    val buffer = ByteArray(4 * 1024)
                    var read: Int
                    while (inputStream.read(buffer).also { read = it } != -1) {
                        outputStream.write(buffer, 0, read)
                    }
                    outputStream.flush()
                }
                return file.absolutePath
            }
        }

        /// モデルと画像をロード
        val module = Module.load(assetFilePath(this, "CNNModel.pt"))

        // 推論ボタンクリック
        inferBtn.setOnClickListener {
            //描いた画像(bitmapを取得)
            val bitmap = drawSurfaceView!!.prevBitmap!!
            // 作成した学習済みモデルの入力サイズにリサイズ
            val bitmapResized    = Bitmap.createScaledBitmap(bitmap,28, 28, true)

            /// テンソル変換と標準化
            val inputTensor = TensorImageUtils.bitmapToFloat32Tensor(
                bitmapResized,
                TensorImageUtils.TORCHVISION_NORM_MEAN_RGB, TensorImageUtils.TORCHVISION_NORM_STD_RGB
            )

            /// 推論とその結果
            /// フォワードプロパゲーション
            val outputTensor = module.forward(IValue.from(inputTensor)).toTensor()
            val scores = outputTensor.dataAsFloatArray

            // リサイズした画像を表示
            sampleImg.setImageBitmap(bitmapResized)

            /// scoreを格納する変数
            // スコアMAXのインデックス = 画像認識で予測した数字 (モデルの作り方から)
            var maxScore: Float = 0F
            var maxScoreIdx = -1
            for (i in scores.indices) {
                Log.d("scores", scores[i].toString()) // スコア一覧をログに出力(どの数字に近いか見てみると面白い)
                if (scores[i] > maxScore) {
                    maxScore = scores[i]
                    maxScoreIdx = i
                }
            }

            // 推論結果を表示
            resultNum.text = "$maxScoreIdx"
        }
    }
}
