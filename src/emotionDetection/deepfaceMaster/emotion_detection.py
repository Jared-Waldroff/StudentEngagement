from deepface.DeepFace import analyze

# Path to your image
img_path = "/Users/Jared.Waldroff/PycharmProjects/StudentEngagement/assets/photo/woman2.png"  # Replace with your actual image path

# Analyze the image for emotions
result = analyze(img_path=img_path, actions=['emotion'])

# Check if result is a list and access the first element
if isinstance(result, list) and len(result) > 0:
    emotion_data = result[0]["emotion"]
    print("Detected Emotion Data:")
    for emotion, confidence in emotion_data.items():
        print(f"{emotion}: {confidence:.2f}")
else:
    print("No face detected or analysis failed.")