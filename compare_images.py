import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.preprocessing import image as keras_image
from keras.models import Model

def compare_images(image_paths):
    features_list, similarities = [], []
    
    for path in image_paths:
        # Load and preprocess image
        img = keras_image.load_img(path, target_size=(224, 224))
        img_data = keras_image.img_to_array(img)
        img_data = np.expand_dims(img_data, axis=0)
        img_data = preprocess_input(img_data)
        
        # Extract VGG16 features
        base_model = VGG16(weights='imagenet')
        model = Model(inputs=base_model.input, outputs=base_model.get_layer('fc1').output)
        features = model.predict(img_data).flatten()
        
        features_list.append(features)
    
    # Calculate similarities
    for i in range(len(features_list)): 
        for j in range(i + 1, len(features_list)): 
            similarity = cosine_similarity(features_list[i].reshape(1, -1), features_list[j].reshape(1, -1))[0][0]
            similarities.append(similarity)

    # Print similarity results
    print("\n--- Similarity Results ---")
    idx = 0
    for i in range(len(image_paths)): 
        for j in range(i + 1, len(image_paths)): 
            print(f"Similarity between image {i+1} and image {j+1}: {similarities[idx]:.4f}")
            idx += 1

def main():
    image_paths = [
        "images/0.jpg",
        "images/0.jpg",
        "images/1.jpg",
        "images/1.jpg",
        "images/2.jpg",
        "images/3.jpg"
    ]
    compare_images(image_paths)

if __name__ == "__main__":
    main()
