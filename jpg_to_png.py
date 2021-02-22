"""
Auteur: Jean-Philippe Langelier 
Étudiant à la maitrise 
Université Laval
"""

from PIL import Image
from IPython.display import display
import PIL.ImageOps

im1 = Image.open('/Users/JP/Documents/Crosser-le-Maire/test_old_bingo/150489310_184764989655020_7577705555349818844_n.jpg')

im1 = im1.resize((300,340))
box = (0, 0, 28, 28)
box2 = (0, 112, 28, 140)
box3 = (112, 0, 140, 28)
box4 = (0, 28, 28, 56)
box5 = (14, 56, 28, 84)
box6 = (100, 89, 114, 115)
box8 = (190,8,205,28)
cropped_image8 = im1.crop(box8).resize((28,28))
cropped_image = im1.crop(box).resize((28,28))
cropped_image2 = im1.crop(box2).resize((28,28))
cropped_image3 = im1.crop(box3).resize((28,28))
cropped_image4 = im1.crop(box4).resize((28,28))
cropped_image5 = im1.crop(box5).resize((28,28))
cropped_image6 = im1.crop(box6).resize((28,28))
i8 = PIL.ImageOps.invert(cropped_image8)
i1 = PIL.ImageOps.invert(cropped_image)
i2 = PIL.ImageOps.invert(cropped_image2)
i3 = PIL.ImageOps.invert(cropped_image3)
i4 = PIL.ImageOps.invert(cropped_image4)
i5 = PIL.ImageOps.invert(cropped_image5)
i6 = PIL.ImageOps.invert(cropped_image6)
display(i8)
display(i1)
display(i2)
display(i3)
display(i4)
display(i5)
display(i6)

i8.save('/Users/JP/Documents/Crosser-le-Maire/test_old_bingo/test_6_2/xxt.png')
i1.save('/Users/JP/Documents/Crosser-le-Maire/test_old_bingo/test_6_2/xxx.png')
i2.save('/Users/JP/Documents/Crosser-le-Maire/test_old_bingo/test_6_2/xxy.png')
i4.save('/Users/JP/Documents/Crosser-le-Maire/test_old_bingo/test_6_2/xxz.png')
i5.save('/Users/JP/Documents/Crosser-le-Maire/test_old_bingo/test_6_2/xxw.png')
i6.save('/Users/JP/Documents/Crosser-le-Maire/test_old_bingo/test_6_2/xxv.png')
