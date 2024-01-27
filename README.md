# Find someone who looks like using facenet
Facenet makes embedding vector and we compare between vectors to find a similar person</br>
You can give a face image to web application, then web application shows someone who looks like face image
# How to use?
1. install
   ```bash
   # clone this repo
   git clone https://github.com/leejeongwoo1/facenet-web-project.git  
   ```
   ```
   check requirements.txt and install libraries(ex. facenet-pytorch, flask ...)
   ```
2. include dataset:
   ```
   Add face image at frontend/static/face_dataset to compare with client image
   ```
3. make embedding vector (json) about face_dataset
   ```
   Run embeddings.py to make json file which includes 512D embedding vector about dataset
   You can check json file at /model 
   ```
4. Run flask web application
   ```shell
   set FLASK_APP=frontend
   flask run
   ```
# reference
[facenet-pytorch](https://github.com/timesler/facenet-pytorch)</br>
[facenet](https://github.com/davidsandberg/facenet)</br>
[lovelyzDetector using facenet](https://github.com/hayunjong83/lovelyzDetector)</br>
# web preview
main page
<img src="./ref/main_page.png" style="zoom: 100%;"/>
find similary person
<img src="./ref/detect_face.png" style="zoom: 100%;">
error page
<img src="./ref/error_page.png" style="zoom: 100%;">
