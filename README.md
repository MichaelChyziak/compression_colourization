# compression_colourization

In summary, this project takes a 250x250 image and compresses it into greyscale along with 128 extra bytes of information. Then it is decompressed using a neural network to re-create the original image.
Further detailed information can be seen in the .pdf file.

# Demo

In order to run the demo, download the neural network model from the following link and place in the code/ directory:
https://www.dropbox.com/s/qt3k2s6tdhqzuvw/my_model.h5?dl=0


To run a quick demo, after installing all dependencies run (inside the code/ directory):
(Any 250x250 image can replace michael.png)
```
python encode.py michael.png
python decode.py
```

The final file will be in decoded/ and the compressed data is saved in encoded/

All *_demo/ directories are only for visual demonstration purposes
