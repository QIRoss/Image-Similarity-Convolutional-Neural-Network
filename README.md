# Image Similarity Convolutional Neural Network

This project allows you to calculate the similarity between images of watches using a pre-trained CNN (Convolutional Neural Network) VGG16 model. The similarities are calculated based on features extracted from the images using cosine similarity.

## Project Structure

- `images/`: Folder containing images of watches (e.g., `0.jpg`, `1.jpg`, `2.jpg`, ..., `20.jpg`).
- `compare_images.py`: Python script containing the logic to calculate image similarities.
- `recommendation_env/`: Python virtual environment for running the project.

## Requirements

You will need the following Python libraries to run the project:

- `Python 3.11`
- `numpy`
- `keras`
- `scikit-learn`
- `matplotlib`

You can install the required dependencies by running:

```bash
pip install -r requirements.txt
```

## Instructions to Run

1. **Set up the environment**:
   - If you don't have the `recommendation_env` already set up, create it using:
     ```
     python -m venv recommendation_env
     ```
   - Activate the virtual environment:
     - On Windows:
       ```
       recommendation_env\Scripts\activate
       ```
     - On macOS/Linux:
       ```
       source recommendation_env/bin/activate
       ```

2. **Install dependencies**:
   - Install the necessary Python libraries:
     ```
     pip install -r requirements.txt
     ```

3. **Download the dataset**:
   - You can download the watch images from the Kaggle dataset [here](https://www.kaggle.com/datasets/mathewkouch/a-dataset-of-watches).
   - Place the downloaded images in the `images/` folder in your project directory.

4. **Run the script**:
   - To calculate the similarities between images, run the following command:
     ```
     python compare_images.py
     ```

   - The script will display similarity results between each pair of images. The results will be printed in the console in the format:
     ```
     Similarity between image 1 and image 2: 0.8542
     Similarity between image 1 and image 3: 0.9014
     ...
     ```
   - Go to ```compare_images.py``` file last lines and change the ```main``` function to test with other images locally or import ```compare_images``` as a module. 
   - You can use the same image multiple times to check that 100% works.
   - Use this idea and set a percentage threshold in a sales system to decide which products to recommend.

---

### Notes

- Ensure the image files are correctly named and placed in the `images/` folder before running the script.
- If you have more images, just add their paths to the `image_paths` list in the script.
