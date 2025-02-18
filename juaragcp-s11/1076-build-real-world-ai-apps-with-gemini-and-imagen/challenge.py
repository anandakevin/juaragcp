import os
from google.cloud import storage
from vertexai.preview.vision_models import ImageGenerationModel, Image
from vertexai.preview.language_models import TextGenerationModel

# Constants
PROJECT_ID = "qwiklabs-gcp-00-ebe03b6b5ec9"  # Replace with your Google Cloud project ID
REGION = "europe-west1"  # Replace with your desired region
BUCKET_NAME = "your-bucket-name"  # Replace with your Google Cloud Storage bucket name

# Set Google Application Credentials (if not already set)
if "GOOGLE_APPLICATION_CREDENTIALS" not in os.environ:
    os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/service-account-key.json"  # Replace with your service account key path

# Initialize Vertex AI
image_generation_model = ImageGenerationModel.from_pretrained("imagegeneration@002")
text_generation_model = TextGenerationModel.from_pretrained("gemini-pro-vision")

def get_storage_client():
    """
    Returns a Google Cloud Storage client.
    Initializes the client only when needed.
    """
    return storage.Client(project=PROJECT_ID)

def generate_bouquet_image(prompt, storage_type="local"):
    """
    Generates an image using the provided prompt and stores it locally or in Google Cloud Storage.
    """
    response = image_generation_model.generate_images(
        prompt=prompt,
        number_of_images=1,
    )
    image = response.images[0]
    
    if storage_type == "local":
        image_path = "bouquet_image.png"
        image.save(image_path)
    elif storage_type == "s3":
        # Initialize storage client only when using S3
        storage_client = get_storage_client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob("bouquet_image.png")
        image.save(blob)
        image_path = f"gs://{BUCKET_NAME}/bouquet_image.png"
    else:
        raise ValueError("Invalid storage_type. Use 'local' or 's3'.")
    
    return image_path

def analyze_bouquet_image(image_path, prompt, storage_type="local"):
    """
    Analyzes the bouquet image and generates birthday wishes using the Gemini Pro Vision model.
    """
    if storage_type == "local":
        image = Image.load_from_file(image_path)
    elif storage_type == "s3":
        # Initialize storage client only when using S3
        storage_client = get_storage_client()
        bucket = storage_client.bucket(BUCKET_NAME)
        blob = bucket.blob(image_path.split("/")[-1])
        image = Image.load_from_bytes(blob.download_as_bytes())
    else:
        raise ValueError("Invalid storage_type. Use 'local' or 's3'.")
    
    response = text_generation_model.generate_text(
        prompt=prompt,
        image=image,
        stream=True,
    )
    
    birthday_wishes = ""
    for chunk in response:
        birthday_wishes += chunk.text
    
    return birthday_wishes

# Example usage
if __name__ == "__main__":
    prompt = "Create an image containing a bouquet of 2 sunflowers and 3 roses"
    image_path = generate_bouquet_image(prompt, storage_type="local")  # Change to "s3" for Google Cloud Storage
    
    birthday_prompt = "Generate birthday wishes based on the bouquet image."
    wishes = analyze_bouquet_image(image_path, birthday_prompt, storage_type="local")  # Change to "s3" for Google Cloud Storage
    print("Birthday Wishes:", wishes)