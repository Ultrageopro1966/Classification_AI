from PIL import ImageDraw, Image
import random
lines = []

for i in range(1200):
    img = Image.new("RGB", (80, 80), (0, 0, 0)) # Создание нового изображения
    draw = ImageDraw.Draw(img, mode = None)
    line = False
    if random.randint(0, 2) != 0:
        for _ in range(50):
            draw.point((random.randint(0, 79), random.randint(0, 79)),
                       (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))) # Отрисовка точек
        for _ in range(0, random.randint(0, 4)):
            draw.line((random.randint(0, 79), random.randint(0, 79), random.randint(0, 79), random.randint(0, 79)), 
                      (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))) # Отрисовка линий
        
        if random.randint(0, 1) == 1:
            radius = random.randint(3, 20)
            x = random.randint(radius, 80 - radius)
            y = random.randint(radius, 80 - radius)
            draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                         (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))) # Отрисовка круга
            line = True

    else:
        for k in range(80):
            for j in range(80):
                color = random.randint(0, 255)
                img.putpixel((k, j), (color, color, color)) # Отрисовка помех
    
    lines.append("1\n" if line else "-1\n")
    img.save(f"dataset/test{i}.png")

# Запись правильных ответов в файл
with open("dataset/data_answers.txt", "w") as f:
    f.writelines(lines)