import os
import zipfile
from PIL import Image, ImageDraw

doc = r"C:\Users\Shravni Morkhade\Downloads\Capstone_Project_Report_CAP_26_FINAL.docx"
out = r"C:\Users\Shravni Morkhade\OneDrive\Documents\ChatGPT\GEN AI\report_media"
os.makedirs(out, exist_ok=True)
items = []
with zipfile.ZipFile(doc) as z:
    for name in z.namelist():
        if name.startswith("word/media/"):
            path = os.path.join(out, os.path.basename(name))
            with open(path, "wb") as f:
                f.write(z.read(name))
            im = Image.open(path).convert("RGB")
            im.thumbnail((260, 180))
            items.append((name, im.copy()))
sheet = Image.new("RGB", (600, ((len(items) + 1) // 2) * 220), "white")
d = ImageDraw.Draw(sheet)
for i, (name, im) in enumerate(items):
    x, y = (i % 2) * 300, (i // 2) * 220
    sheet.paste(im, (x, y + 20))
    d.text((x, y), name, fill="black")
sheet.save(os.path.join(out, "contact.png"))
