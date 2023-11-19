from PIL import Image

img = Image.open('raton.jpg')
imgGray = img.convert('L')
imgGray.save('raton_gray.jpg')
