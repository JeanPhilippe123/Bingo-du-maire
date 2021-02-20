"""
Auteur: Jean-Philippe Langelier 
Étudiant à la maitrise 
Université Laval
"""

from PIL import Image
from IPython.display import display

im1 = Image.open('/Users/JP/Documents/Bingo/test_old_bingo/150489310_184764989655020_7577705555349818844_n.jpg')

im1 = im1.resize((300,340))
box = (0, 0, 28, 28)
box2 = (0, 112, 28, 140)
box3 = (112, 0, 140, 28)
box4 = (0, 28, 28, 56)
box5 = (14, 56, 28, 84)
box6 = (100, 89, 114, 115)
cropped_image = im1.crop(box)
cropped_image2 = im1.crop(box2)
cropped_image3 = im1.crop(box3)
cropped_image4 = im1.crop(box4)
cropped_image5 = im1.crop(box5).resize((28,28))
cropped_image6 = im1.crop(box6).resize((28,28))
# print(cropped_image)
display(cropped_image)
display(cropped_image2)
display(cropped_image3)
display(cropped_image4)
display(cropped_image5)
display(cropped_image6)

cropped_image.save('/Users/JP/Documents/Bingo/test_old_bingo/test_6_2/xxx.png')
cropped_image2.save('/Users/JP/Documents/Bingo/test_old_bingo/test_6_2/xxy.png')
cropped_image4.save('/Users/JP/Documents/Bingo/test_old_bingo/test_6_2/xxz.png')
cropped_image5.save('/Users/JP/Documents/Bingo/test_old_bingo/test_6_2/xxw.png')
cropped_image6.save('/Users/JP/Documents/Bingo/test_old_bingo/test_6_2/xxv.png')
