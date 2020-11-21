import os

from flask import Flask, render_template, request, redirect
from inference import get_prediction,get_perb
from commons import format_class_name
from PIL import Image
import io
import base64
import torchvision.transforms as transforms

app = Flask(__name__)


@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files.get('file')
        if not file:
            return render_template('index.html')

        img_bytes = file.read()


        modelname = request.form.get('modelname')
        basemodel = request.form.get('basename')

        class_id, class_name = get_prediction(image_bytes=img_bytes,modelname=modelname)
        class_name = format_class_name(class_name)
        # image = Image.open(io.BytesIO(img_bytes))
        [class_idp, class_namep], stimg, perbimg,st_perb,noise = get_perb(image_bytes=img_bytes,modelname=modelname,basemodel=basemodel)
        
        class_namep = format_class_name(class_namep)

        my_transforms = transforms.Compose([transforms.Resize(255),
                                        transforms.CenterCrop(224),
                                        ])


        img = my_transforms(Image.open(io.BytesIO(img_bytes)))
        buffered = io.BytesIO()
        img.save(buffered, format="png")
        buffered.seek(0)
        img_bytes = buffered.getvalue()


        return render_template('result.html', class_id=class_id,model_name = modelname,base_model = basemodel,
                               class_name=class_name,ori_image="data:image/png;base64," + base64.b64encode(img_bytes).decode(),class_idp=class_idp,
                               class_namep=class_namep,st_image ="data:image/png;base64," + base64.b64encode(stimg).decode(),
                                perb_image="data:image/png;base64," + base64.b64encode(perbimg).decode(),noise_image= "data:image/png;base64," + base64.b64encode(noise).decode(),st_perb = "data:image/png;base64," + base64.b64encode(st_perb).decode())
    return render_template('index.html')


if __name__ == '__main__':
    app.run(debug=True, port=int(os.environ.get('PORT', 5000)))
