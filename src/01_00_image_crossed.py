from PIL import Image, ImageDraw

image = Image.open("panda-corner.jpg")

print(image.format) # Output: JPEG
print(image.mode) # Output: RGB
print(image.palette) # Output: None

image.show()

draw = ImageDraw.Draw(image)
draw.line((0, 0) + image.size, fill=128)
draw.line((0, image.size[1], image.size[0], 0), fill=128)
del draw

# write to stdout
image.save("out/panda-cross.png", format="PNG")
