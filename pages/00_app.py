import streamlit as st
import numpy as np
import tensorflow as tf
from keras.applications.mobilenet_v2 import preprocess_input
from PIL import Image
import importlib
import requests


def prediction(modelname,sample_image,IMG_SIZE=(224,224)):
    #labels
    labels = ["ADONIS","NormalAFRICAN_GIANT_SWALLOWTAIL","AMERICAN_SNOOT","AN_88","APPOLLO","ATALA","BANDED_ORANGE_HELICONIAN",
    "BANDED_PEACOCK","BECKERS_WHITE","BLACK_HAIRSTREAK","BLUE_MORPHO","BLUE_SPOTTED_CROW","BROWN_SIPROETA","CABBAGE_WHITE","CAIRNS_BIRDWING",
    "CHECQUERED_SKIPPER","CHESTNUT","CLEOPATRA","CLODIUS_PARNASSIAN","CLOUDED_SULPHUR","COMMON_BANDED_AWL","COMMON_WOOD-NYMPH","COPPER_TAIL",
    "CRECENT","CRIMSON_PATCH","DANAID_EGGFLY","EASTERN_COMA","EASTERN_DAPPLE_WHITE","EASTERN_PINE_ELFIN","ELBOWED_PIERROT","GOLD_BANDED",
    "GREAT_EGGFLY","GREAT_JAY","GREEN_CELLED_CATTLEHEART","GREY_HAIRSTREAK","INDRA_SWALLOW","IPHICLUS_SISTER","JULIA","LARGE_MARBLE",
    "MALACHITE","MANGROVE_SKIPPER","MESTRA","METALMARK","MILBERTS_TORTOISESHELL","MONARCH","MOURNING_CLOAK","ORANGE_OAKLEAF",
    "ORANGE_TIP","ORCHARD_SWALLOW","PAINTED_LADY","PAPER_KITE","PEACOCK","PINE_WHITE","PIPEVINE_SWALLOW","POPINJAY","PURPLE_HAIRSTREAK",
    "PURPLISH_COPPER","QUESTION_MARK","RED_ADMIRAL","RED_CRACKER","RED_POSTMAN","RED_SPOTTED_PURPLE","SCARCE_SWALLOW","SILVER_SPOT_SKIPPER",
    "SLEEPY_ORANGE","SOOTYWING","SOUTHERN_DOGFACE","STRAITED_QUEEN","TROPICAL_LEAFWING","TWO_BARRED_FLASHER","ULYSES","VICEROY",
    "WOOD_SATYR","YELLOW_SWALLOW_TAIL"] #35 labels

    try:
        #loading the .h5 model
        load_model = tf.keras.models.load_model(modelname)

        sample_image = Image.open(sample_image)
        img_array = sample_image.resize(IMG_SIZE)
        img_batch = np.expand_dims(img_array, axis = 0) # Rows
        image_batch = img_batch.astype(np.float32)
        image_batch = preprocess_input(image_batch)
        prediction = load_model.predict(img_batch)
        return labels[int(np.argmax(prediction, axis = 1))] # Columns


    except Exception as e:
        st.write("ERROR: {}".format(str(e)))

    
st.title=("H5 Model Butterfly Classifier")

st.image(
    "https://img.freepik.com/free-photo/fantasy-landscape-with-butterfly_23-2151451739.jpg", 
    caption = "Butterfly")

#about the web app
st.header("About the Web App")

#details about the project
with st.expander("Web App 🌐"):
    st.subheader("Butterfly Image Predictions")
    st.write("This web app is about.....................")


#setting the tabs
tab1, tab2 = st.tabs(['Image Upload 👁️', 'Camera Upload 📷'])


#tab1
with tab1:
    #setting file uploader
    #you can change the label name as your preference
    image = st.file_uploader(label="Upload an image",accept_multiple_files=False, help="Upload an image to classify them")

    if image:
        #validating the image type
        image_type = image.type.split("/")[-1]
        if image_type not in ['jpg','jpeg','png','jfif']:
            st.error("Invalid file type : {}".format(image.type), icon="🚨")
        else:
            #displaying the image
            st.image(image, caption = "Uploaded Image")

            #getting the predictions
            label = prediction("best_model_saved.h5", image)

            #displaying the predicted label
            st.subheader("Your Classification is **{}**".format(label))

with tab2:
    #camera input
    cam_image = st.camera_input("Please take a photo")

    if cam_image:
        #displaying the image
        st.image(cam_image)

        #getting the predictions
        label = prediction("best_model_saved.h5", cam_image)

        #displaying the predicted label
        st.subheader("Your Classification is **{}**".format(label))


