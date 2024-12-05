from flask import Flask, request, jsonify, send_file
from flask_cors import CORS, cross_origin
from ultralytics import YOLO
from waitress import serve
import os

class App:
    def __init__(self):
        self.app = Flask(__name__)
        CORS(self.app)
        print("App initializing...")
        print("loading model...")
        self.model = YOLO("phone_detectv2.pt")
        print("model loaded...")
        self.app.add_url_rule('/predict/heathcheck', view_func=self.api, methods=['GET'])
        self.app.add_url_rule('/predict', view_func=self.predict, methods=['POST'])
        self.app.add_url_rule('/predict/image', view_func=self.get_image_predict, methods=['GET'])
        self.app.add_url_rule('/predict/result', view_func=self.get_results, methods=['GET'])
        self.warm_up()
        print("App initialized...")

    def api(self):
        return jsonify({"message": "Hello World!"})
    
    def warm_up(self):
        print("Warming up...")
        self.predict_image("./dummy.jpg")
        print("Warmed up...")

    def predict(self):
        conf = request.args.get('conf', default=0.4, type=float)
        if 'file' not in request.files:
            return jsonify({"message": "No file part"})

        files = request.files.getlist('file')
        if not files:
            return jsonify({"message": "No selected files"})

        output_directory = os.path.join("images")

        if not os.path.exists(output_directory):
            os.makedirs(output_directory)

        results = []
        for file in files:
            if file.filename == '':
                results.append({"filename": "", "message": "No selected file"})
                continue

            file_path = os.path.join(output_directory, file.filename)
            file.save(file_path)
            res_path = os.path.abspath(f"results_{file.filename}.txt")

            try:
                res = self.predict_image(file_path, conf=conf, save_path=res_path)
                if res == 0:
                    results.append({"filename": file.filename, "message": "Success", "code": 0})
                elif res == 1:
                    results.append({"filename": file.filename, "message": "No object detected", "code": 1})
                else:
                    results.append({"filename": file.filename, "message": "Error", "code": 2})
            except Exception as e:
                results.append({"filename": file.filename, "message": str(e), "code": 2})

        return jsonify(results)
    
    def predict_image(self, image, conf=0.25, save_path="results.txt"):
        try:
            # remove if file exists
            if os.path.exists(save_path):
                os.remove(save_path)
            
            # Create an empty file
            with open(save_path, 'w') as f:
                pass

            res = self.model(image, conf=conf, save=False)
            res[0].save_txt(txt_file=save_path)
            res[0].save(save_path.replace(".txt", ".jpg"))

            # check value in file, if not exist return None
            with open(save_path, "r") as f:
                content = f.read()
                if content:
                    return 0 # success
                else:
                    return 1 # no object detected
        except Exception as e:
            print(e)
            return 2 # error
    
    def get_image_predict(self):
        try:
            res_path = os.path.abspath("results.txt")
            return send_file(res_path, as_attachment=True)
        except Exception as e:
            return jsonify({"message": str(e)})
    
    def get_results(self):
        res_path = os.path.abspath("results.txt")
        with open(res_path, "r") as f:
            return f.read()

    def run(self):
        serve(self.app, host='0.0.0.0', port=5001)  # Specify the host and port for waitress
