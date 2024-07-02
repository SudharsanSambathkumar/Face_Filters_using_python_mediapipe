# Face_Filters_using_python_mediapipe

This project applies various face filters using computer vision techniques. The filters include masks and accessories from popular superhero characters like Deadpool, Wolverine, and Captain America.

## Project Structure
- `main.ipynb`: Jupyter Notebook containing the main code for applying face filters.
- `main.py`: Python script containing the main code for applying face filters.
- `images/`: Directory containing image assets used for the filters.
  - `claw_1.png`: Wolverine claw image.
  - `deadpool1.png`: Deadpool mask image.
  - `deadpool2.png`: Another Deadpool mask image.
  - `shield_1.png`: Captain America's shield image.
  - `Wolverine_3.png`: Wolverine mask image.

## How to Run
1. Clone the repository:
```bash
git clone https://github.com/your-username/face-filters.git
cd face-filters
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

3. Run the Jupyter Notebook:
```bash
jupyter notebook main.ipynb
```
   OR run the Python script:
```bash
python main.py
```

4. Follow the instructions in `main.ipynb` or `main.py` to see the face filters applied to sample images.

## Description
This project demonstrates how to use computer vision to overlay various filters on faces. By utilizing OpenCV and other Python libraries, the project can detect faces in an image and superimpose fun and thematic filters based on popular superhero characters.

## Sample Filters
- **Deadpool Mask**: Adds Deadpool's mask over the face.
- **Wolverine Mask and Claws**: Adds Wolverine's mask and claws to the face and hands.
- **Captain America's Shield**: Adds Captain America's shield.
