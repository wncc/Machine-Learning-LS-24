# Image Preprocessing - OpenCV
Image preprocessing using OpenCV is crucial for deep learning projects as it enhances image quality, ensuring better model accuracy. Techniques include resizing, normalization, noise reduction, and augmentation. These steps standardize inputs, improve feature extraction, and increase dataset diversity, leading to more robust and efficient neural network training and performance.

Frameworks like Tensorflow also have image preprocessing methods but aren't as efficient as OpenCV (made for computer vision only). You can use framework's preprocessing methods like resizing, normalization, rotation etc. for simple image data. But for advanced image preprocessing techniques you need OpenCV.

>[!Tip]
All the videos can be watched at **2x speed**

## Image Basics

In the context of computer vision and deep learning, an image represented as a NumPy array is a multi-dimensional array where each element corresponds to a pixel value. Each pixel's value ranges from 0 to 255, indicating the intensity of the color. This representation allows for efficient manipulation and processing using NumPy's array operations.

[What is Image?](https://youtu.be/oUJs03eZ0S8?si=tyYuoZIt091fhbYp)  
[Image Reading](https://youtu.be/wRtAoZF50Jc?si=m7RWW3Qy7pQxYaed)

## Image Color Channels
 For a grayscale image, it's a 2D array with dimensions (height, width), and for a color image, it's a 3D array with dimensions (height, width, channels), where channels typically represent RGB values.
>[!Note]
>OpenCV reads image as BGR not RGB

[BGR2GRAY](https://youtu.be/AFrZ3JOQ0Qg?si=wcdHgOTnidg7bHYV)  
[BGR Channels](https://youtu.be/wlH9w1eA6PQ?si=AFraoDNEW-3YYJB1)  
[BGR vs RGB](https://youtu.be/kSqxn6zGE0c?si=_ZK8MVWV5SLiJWi1&t=581) (9:41 to 11:50)

## Playing with Images

Image cropping, resizing, rotation and flipping are necessary for standardizing input sizes, enhancing model training, and augmenting data to improve the robustness and performance of deep learning models. Drawing bounding boxes is essential for object detection projects.

[Resizing](https://youtu.be/DPkpI2ezVO4?si=-xlW6J5h0TW8Fnh5)  
[Flipping](https://youtu.be/Y_78ARbpSwo?si=iCVqHqxZdX-3-lCp)  
[Cropping](https://youtu.be/fanEPKLRbPk?si=E4yGRKhyIByJq7ov)  
[Rotation](https://youtu.be/MtHvL1emJSE?si=uzGlA-SX9vedrJfL)  
[Drawing Shapes](https://youtu.be/shfXj_Og7ak?si=Yv7_qiBL7IVulHQl)

## Blurring and Noise Removal

Blurring and noise removal are essential preprocessing steps in image processing. Blurring, using techniques like Gaussian blur, smooths the image, reducing detail and noise. Noise removal enhances image quality by eliminating random variations, ensuring clearer, more accurate inputs for deep learning models.

[Blurring](https://youtu.be/eDIj5LuIL4A?si=uiqeiB6PriZciqRz&t=3099) (51:39 to 1:07:00)

## Threshold and Edge Detection

Thresholding converts grayscale images to binary by setting pixel intensity limits, simplifying analysis. Edge detection, using methods like Canny or Sobel, identifies boundaries within images, highlighting significant structural information essential for feature extraction in deep learning models.

[Threshold and Edge Detection](https://youtu.be/eDIj5LuIL4A?si=RtmvrTYhClNw_6b4&t=4026) (1:07:06 to 1:31:30)

## Contours

**Contours:**  
Contours are *curves joining continuous points along the boundary of an object* in an image. They represent the shape and structure of objects present in the image, making them useful for tasks such as object detection, shape analysis, and recognition.

**Edge Detection:**  
Edge detection algorithms aim to identify points in an image where the *brightness changes abruptly*. These points typically correspond to boundaries between objects or significant changes in texture within the image.

[Contours Detection](https://youtu.be/eDIj5LuIL4A?si=-FRfiF9eB5D0z--_&t=6317) (1:45:17 to 2:01:20)

## Image Saving

It's very important to save the images after processing them.

[Image Saving](https://youtu.be/b_vVNCVDrbw?si=iaix9mp4M9mSzzi6)
