<div align="center">
<img src="./assets/icon.svg" width="100" height="100" />
<h1>plasma</h1>
</div>

Idk i was kinda having problems with writing circuits again and again for torch. Note that this is not
the rewriting of anything anywhere and is merely a wrapper. I am wholely bound by Qiskit for
the actual bottleneck in speed.

**This for better DX and not more speed**

## Design Choices
### Pooling
There are 3 kinds of pooling generally used; max, min and average

**MaxPooling**: is generally used for classification problems where we expect brightness
to pull out the edges of the image. In such a situation if we use average pooling
it will smear out the edges and make it harder to classify.
Since the MNIST dataset is a white digit on a black background, we use max pooling
to pull out the edges of the digit. [See More](https://medium.com/@bdhuma/which-pooling-method-is-better-maxpooling-vs-minpooling-vs-average-pooling-95fb03f45a9)