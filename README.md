# HDR Envmap 合成以及太阳亮度校正

### 亮度单位

由于需要校正亮度，所以需要定下一个亮度的参考单位。

亮度参考暂定为使用 Sony 相机，ISO * exposure_time (s) = 1，光圈为 f/3.5 时，不考虑饱和时摄得的强度（设该设置下饱和时的强度为1）。
Sony 相机本身拍摄的图像容易使用该亮度表示，可直接通过拍摄设置计算。
Ricoh 相机拍摄的图像若要转为该亮度表示，可以先转为 Ricoh 相机本身 ISO * exposure_time (s) = 1，光圈为 f/3.5 时的强度，再使用标定的系数转换到 Sony 亮度空间。

在处理过程中，HDR 图像文件中的强度应以该强度为单位进行存储。
处理完成后或许可以对一整组图像（包括HDR环境光照及人脸图像）进行一致的亮度调整。

### 拍摄前的准备工作

标定两个颜色校正矩阵（Color Correction Matrix，CCM）：

1. Ricoh 的 Camera Response Function (CRF) 与 Sony 的 CRF 之间的 CCM。这是由于拍人脸时使用的是 Sony 相机，而拍环境光照时使用的是 Ricoh 相机，它们的 CRF 不同，所以即使拍摄同样物体，Raw Image 的值也会有差异。为了尽量去除该差异，需要标定它们之间的 CCM。
   标定方法：用 Sony 相机和 Ricoh 相机分别拍摄同一场景的同一 ColorChecker，然后根据 Raw Image 用最小二乘标定。（已标定，```./Data/results/ricoh2sony_mat.npy```）
2. ND Filter 3.0 的 CCM。这是由于拍太阳时需要用到 ND Filter，而它并不像它的名字那样一致地滤去所有波长的光，而是会带来色偏。为了去除色偏以及标定它具体的过滤系数，需要进行标定。
   标定方法：用 Sony 相机在带 Filter 和不带 Filter 设置下拍摄两次同一场景同一 ColorChecker，然后根据 Raw Image 用最小二乘标定。（已拍摄，未标定）

### HDR Envmap 合成步骤

1. （Python）处理 Ricoh 摄得的 .DNG 原始图像文件，应用 CCM 使得其颜色变为使用 Sony 相机拍摄时该有的颜色。
   1. 使用 txt 文件记录每一图像的曝光时间、ISO 及光圈，后面合成时使用。
2. （Python）导出为 TIFF 文件（.TIF），以待 RICOH THETA Stitcher 处理。
3. （手动）在 RICOH THETA Stitcher 中将 TIFF 格式的双鱼眼图像拼接为360°全景图像。
   1. 在拼接时，调整 Pitch 和 Roll 使得画面中的 Sony 相机（拍摄人脸用。实际拍摄时，Ricoh 相机应与人脸先后摆在同一位置）处于全景图的正中心。
   2. 应用同样的几何校正设置处理一整组全景图像。输出图像后缀会自动设为 _er.tif。
4. （Python）对拼接后的全景图像进行 Fusion，输出为标准亮度单位下的 .HDR 或者 .EXR 文件。
   1. 注意此时的太阳亮度是 Clipped 的，颜色及强度均不正确。

### 太阳亮度校正

在拍摄时，应该使用装了 ND Filter 3.0 的 Sony 相机对太阳进行拍摄，使其在图像上的亮度未饱和。

然后考虑将 Sony 相机记录的太阳亮度迁移到 Ricoh 相机中。

尝试思路：
认为亮度（即输入能量除以立体角）×立体角为不变量，对太阳亮度进行迁移。
在 Sony 相机已知的拍摄设置下（焦距、光圈、传感器大小等），计算每一像素 $(x,y)$ 对应的立体角为 $SR_{sony}(x,y)$，读取亮度为 $I(x, y)$，然后计算总亮度为
$$
E_{sun}=\sum_{(x,y)\in \text{Sun Area}} SR_{sony}(x,y)\cdot I(x,y)
$$
然后迁移到 Ricoh 全景环境光照图像中，计算每一像素的立体角$SR_{ricoh}(x,y)$，把太阳区域的亮度设置为
$$
\frac{E_{sun}}{\sum_{(x,y)\in \text{Sun Area}} SR_{ricoh}(x,y)}.
$$
这样就迁移完毕了。
这个思路从原理上对吗？