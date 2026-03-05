from flask import Flask, render_template,request,jsonify,redirect,url_for,session
import os
import random
import json
from PIL import Image
import requests
from flask import send_from_directory
from flask import request
import numpy as np
import cv2
import mediapipe as mp
import math
import joblib
from pathlib import Path
from werkzeug.utils import secure_filename
from werkzeug.exceptions import RequestEntityTooLarge
import uuid
import re
from dotenv import load_dotenv

try:
    _deepface_module = __import__('deepface')
    DeepFace = getattr(_deepface_module, 'DeepFace', None)
    DEEPFACE_AVAILABLE = DeepFace is not None
except Exception:
    DeepFace = None
    DEEPFACE_AVAILABLE = False

app = Flask(__name__)
app.secret_key = 'your_secret_key'

load_dotenv()

RAPIDAPI_HOST = os.getenv('RAPIDAPI_HOST', 'try-on-diffusion.p.rapidapi.com')
RAPIDAPI_URL = os.getenv('RAPIDAPI_URL', f"https://{RAPIDAPI_HOST}/try-on-file")
RAPIDAPI_KEY = os.getenv('RAPIDAPI_KEY', '').strip()


def rapidapi_headers():
    if not RAPIDAPI_KEY:
        return None
    return {
        'x-rapidapi-host': RAPIDAPI_HOST,
        'x-rapidapi-key': RAPIDAPI_KEY,
    }

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024

CART_FILE = 'cart.json'



with open('data.json', 'r') as jf:
    data = json.load(jf)
data.setdefault('login', {})


def save_login_data():
    with open('data.json', 'w') as f:
        json.dump(data, f, indent=2)


def _load_json_file(path, default_value):
    try:
        with open(path, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return default_value


def _save_json_file(path, payload):
    with open(path, 'w') as f:
        json.dump(payload, f, indent=2)


def load_cart_store():
    return _load_json_file(CART_FILE, {})


def save_cart_store(cart_payload):
    _save_json_file(CART_FILE, cart_payload)


def get_user_key():
    user = session.get('user')
    if not user:
        return None
    return user + str(request.remote_addr)


def normalize_cart_items(items):
    merged = []
    lookup = {}
    for item in items or []:
        key = (
            item.get('img', ''),
            item.get('selected_size', ''),
            item.get('gender', ''),
            item.get('category', ''),
        )
        quantity = int(item.get('quantity', 1) or 1)
        if quantity < 1:
            quantity = 1

        if key in lookup:
            merged[lookup[key]]['quantity'] += quantity
        else:
            normalized = dict(item)
            normalized['quantity'] = quantity
            lookup[key] = len(merged)
            merged.append(normalized)
    return merged


def _format_product_name(filename, category, gender):
    stem = Path(filename).stem.replace('_', ' ').replace('-', ' ').strip()
    stem = re.sub(r'\s+', ' ', stem)
    if stem:
        return stem.title()
    fallback = f"{gender.title()} {category.title()}"
    return fallback


def _generate_products_from_folder(gender, category, image_subfolder, folder_path, base_price, price_step, variants_per_image=1):
    products = []
    if not os.path.isdir(folder_path):
        return products

    files = sorted(
        [
            file_name for file_name in os.listdir(folder_path)
            if file_name.lower().endswith(('.png', '.jpg', '.jpeg'))
        ]
    )

    style_tags = ["Classic", "Premium"]
    running_index = 0
    for idx, file_name in enumerate(files):
        base_name = _format_product_name(file_name, category, gender)
        image_path = f"./static/assets/{'mens' if gender == 'men' else 'women'}/{image_subfolder}/{file_name}"
        collection = 'summer' if idx % 2 == 0 else 'winter'

        for variant in range(variants_per_image):
            tag = style_tags[variant % len(style_tags)]
            product_name = base_name if variants_per_image == 1 else f"{base_name} {tag}"
            price = str(base_price + (running_index * max(40, price_step // 2)))
            products.append({
                "name": product_name,
                "brand": "StyleHub",
                "price": price,
                "img": image_path,
                "sizes": ["S", "M", "L", "XL"],
                "rating": f"{round(4.1 + (running_index % 8) * 0.1, 1)}",
                "reviews": str(12 + running_index * 2),
                "collection": collection,
                "category": category,
                "index": running_index
            })
            running_index += 1
    return products


def _merge_category(existing_items, generated_items, category_name):
    merged = []
    seen_images = set()

    for item in existing_items + generated_items:
        image_path = item.get('img', '')
        if not image_path or image_path in seen_images:
            continue
        seen_images.add(image_path)
        merged_item = dict(item)
        item_name = str(merged_item.get('name', '')).strip()
        if item_name.endswith(' Classic') or item_name.endswith(' Premium'):
            merged_item['name'] = item_name.rsplit(' ', 1)[0]
        merged_item['category'] = category_name
        merged_item.setdefault('collection', 'summer' if len(merged) % 2 == 0 else 'winter')
        merged.append(merged_item)

    for idx, product in enumerate(merged):
        product['index'] = idx
    return merged


def load_dataset():
    dataset = {}
    try:
        with open('dataset.json', 'r') as file:
            dataset = json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        dataset = {}

    dataset.setdefault('men', {})
    dataset.setdefault('women', {})

    men_sources = {
        'tshirt': ('tshirts', 'static/assets/mens/tshirts', 899, 120),
        'shirts': ('shirts', 'static/assets/mens/shirts', 1199, 150),
        'jackets': ('jackets', 'static/assets/mens/jackets', 1899, 220),
    }

    women_sources = {
        'tops': ('tops', 'static/assets/women/tops', 999, 140),
        'jackets': ('jackets', 'static/assets/women/jackets', 1899, 240),
        'dresses': ('casuals', 'static/assets/women/casuals', 1299, 180),
    }

    for category, (image_subfolder, folder, base_price, price_step) in men_sources.items():
        generated = _generate_products_from_folder('men', category, image_subfolder, folder, base_price, price_step, variants_per_image=1)
        existing = dataset.get('men', {}).get(category, [])
        dataset['men'][category] = _merge_category(existing, generated, category)

    for category, (image_subfolder, folder, base_price, price_step) in women_sources.items():
        generated = _generate_products_from_folder('women', category, image_subfolder, folder, base_price, price_step, variants_per_image=1)
        existing = dataset.get('women', {}).get(category, [])
        dataset['women'][category] = _merge_category(existing, generated, category)

    dataset['men'].pop('jeans', None)
    return dataset

@app.route('/get_data', methods=['GET'])
def getData():
    try:
        dataset = load_dataset()
        return jsonify(json.dumps(dataset))
    except Exception:
        return jsonify({"error": "Dataset not found"}), 404

@app.route('/home')
def home():
    return render_template('home.html')

@app.route('/', methods=['GET', 'POST'])
def login():
    error = None
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not email or not password:
            error = 'Email and password are required.'
            return render_template('login.html', error=error, email=email), 400

        if not re.fullmatch(r'[^@\s]+@[^@\s]+\.[^@\s]+', email):
            error = 'Please enter a valid email address.'
            return render_template('login.html', error=error, email=email), 400

        if data['login'].get(email, None) and data['login'][email]==password:
            session['user']=email
            session.permanent = True
            return redirect(url_for('home'))
        error = 'Invalid email or password.'
        return render_template('login.html', error=error, email=email), 401

    return render_template('login.html', error=error)


@app.route('/register', methods=['GET', 'POST'])
def register():
    error = None
    if request.method == 'POST':
        email = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')
        confirmPassword = request.form.get('confirm-password', '')

        if not email or not password or not confirmPassword:
            error = 'All fields are required.'
            return render_template('register.html', error=error, email=email), 400

        if not re.fullmatch(r'[^@\s]+@[^@\s]+\.[^@\s]+', email):
            error = 'Please enter a valid email address.'
            return render_template('register.html', error=error, email=email), 400

        if password != confirmPassword:
            error = 'Passwords do not match.'
            return render_template('register.html', error=error, email=email), 400

        if len(password) < 6:
            error = 'Password must be at least 6 characters.'
            return render_template('register.html', error=error, email=email), 400

        if data['login'].get(email):
            error = 'Account already exists. Please login.'
            return render_template('register.html', error=error, email=email), 409

        data['login'][email] = password
        save_login_data()

        session['user']=email
        session.permanent = True

        return redirect(url_for('home'))

    return render_template('register.html', error=error)


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))



def get_womens_items():

    global womens_images_folder
    womens_images_folder = os.path.join(app.static_folder, 'assets', 'women','tops')

    # Get a list of filenames in the folder
    womens_clothing_images = [file for file in os.listdir(womens_images_folder) if file.endswith(('.jpg', '.png', '.jpeg'))]

    # Create a list of dictionaries with image and random price
    womens_clothing_data = [{'image': file, 'price': random.randint(400, 600)} for file in womens_clothing_images]

    return womens_clothing_data


# @app.route('/women', methods=['GET', 'POST'])
# def women():

#     women_data = get_womens_items()

#     if request.method == 'POST':
#         index = int(request.form.get('index'))
#         uploaded_file = request.files['file']

#         # do processing
#         im = Image.open(os.path.join(womens_images_folder, women_data[index]['image']))
#         im.save('./inputs/cloth.jpg')
#         uploaded_file.save('./inputs/model.jpg')

#         #temp image src , replace with result src
#         image="/static/assets/mens/tshirt-4.png"
#         return jsonify(image)
        

#     return render_template('women.html',women_data=women_data,len=len(women_data))


# Dynamic route for individual product pages
@app.route('/<gender>/<category>/<int:index>', methods=['GET','POST'])
def product_details(gender, category, index):

    print(gender,category,index)
    if request.method == 'POST':
        gender = gender.replace('"','')
        dataset = load_dataset()
        products = dataset.get(gender, {}).get(category, [])

        print(products)
        if 0 <= index < len(products):
            # Retrieve the details of the selected product using the index
            selected_product = products[index]

            # cloth image path
            img = selected_product['img']
            uploaded_img = request.files['file']

            # do processing

            print("Reached Here !")

            #temp image src , replace with result src
            image="/static/assets/mens/shirts/shirt2.png"
            return jsonify(image)
        return jsonify("No products found")
    else:
        print("out of if")
        dataset = load_dataset()
        products = dataset.get(gender, {}).get(category, [])

        if 0 <= index < len(products):
            # Retrieve the details of the selected product using the index
            selected_product = products[index]
            img = selected_product['img'].replace("./static/","")
            return render_template('product_details.html', product=selected_product,gender=gender,img=img,index=index)
        else:
            return render_template('error.html', message=f'Invalid index for {category}')
        

# Assuming cart.json exists and is initially an empty dictionary
cart_data = {}

@app.route('/add_to_cart/<gender>/<category>/<int:index>', methods=['POST'])
def addToCart(gender, category, index):
    try:
        if 'user' not in session:
            return jsonify({"error": "Please login first"}), 401

        dataset = load_dataset()
        products = dataset.get(gender, {}).get(category, [])

        if 0 <= index < len(products):
            # Retrieve the details of the selected product using the index
            selected_product = dict(products[index])

            selected_size = request.form.get('size')
            if not selected_size and request.is_json:
                selected_size = (request.get_json(silent=True) or {}).get('size')

            valid_sizes = {'XS', 'S', 'M', 'L', 'XL'}
            if selected_size not in valid_sizes:
                return jsonify({"error": "Please select a valid size"}), 400

            selected_product['selected_size'] = selected_size
            selected_product['gender'] = gender
            selected_product['category'] = category
            selected_product['quantity'] = 1

            user = session.get('user')
            # Construct the key for the user in the cart
            user_key = user + str(request.remote_addr)

            cart_store = load_cart_store()
            cart_store[user_key] = normalize_cart_items(cart_store.get(user_key, []))

            # If the user does not exist in the cart, create an empty list
            if user_key not in cart_store:
                cart_store[user_key] = []

            existing_item = None
            for item in cart_store[user_key]:
                if (
                    item.get('img') == selected_product.get('img')
                    and item.get('selected_size') == selected_product.get('selected_size')
                    and item.get('gender') == selected_product.get('gender')
                    and item.get('category') == selected_product.get('category')
                ):
                    existing_item = item
                    break

            if existing_item is not None:
                existing_item['quantity'] = int(existing_item.get('quantity', 1) or 1) + 1
                updated_qty = existing_item['quantity']
            else:
                cart_store[user_key].append(selected_product)
                updated_qty = 1

            # Save the updated cart data to cart.json
            save_cart_store(cart_store)

            return jsonify({"message": "Added to Cart", "size": selected_size, "quantity": updated_qty})
        else:
            return jsonify({"error": f'Invalid index for {category}'}), 404
    except FileNotFoundError:
        return jsonify({"error": 'Dataset not found'}), 500




@app.route('/men', methods=['GET','POST'])
def men():
    return render_template('men.html')

@app.route('/women', methods=['GET','POST'])
def women():
    return render_template('women.html')

@app.route('/view-cart', methods=['GET','POST'])
def view_cart():
    if 'user' not in session:
        # If the user is not logged in, redirect them to the login page
        return redirect('/')  # Provide the correct login page URL

    current_user = session['user']
    if request.method == 'GET':
        # Read cart data from cart.json for the current user
        cart_store = load_cart_store()
        user_key = current_user + str(request.remote_addr)
        cart_data = normalize_cart_items(cart_store.get(user_key, []))
        cart_store[user_key] = cart_data
        save_cart_store(cart_store)

        print(cart_data)

        return render_template('view_cart.html', cart_data=cart_data)


@app.route('/cart/quantity', methods=['POST'])
def update_cart_quantity():
    if 'user' not in session:
        return jsonify({'error': 'Please login first'}), 401

    payload = request.get_json(silent=True) or {}
    item_index = payload.get('item_index')
    action = str(payload.get('action', '')).strip()

    if action not in {'inc', 'dec'}:
        return jsonify({'error': 'Invalid action'}), 400

    try:
        item_index = int(item_index)
    except (TypeError, ValueError):
        return jsonify({'error': 'Invalid item index'}), 400

    user_key = get_user_key()
    cart_store = load_cart_store()
    user_cart = normalize_cart_items(cart_store.get(user_key, []))
    cart_store[user_key] = user_cart
    save_cart_store(cart_store)

    if item_index < 0 or item_index >= len(user_cart):
        return jsonify({'error': 'Item not found'}), 404

    item = user_cart[item_index]
    current_qty = int(item.get('quantity', 1) or 1)
    new_qty = current_qty + 1 if action == 'inc' else current_qty - 1

    if new_qty <= 0:
        user_cart.pop(item_index)
    else:
        item['quantity'] = new_qty

    cart_store[user_key] = user_cart
    save_cart_store(cart_store)

    updated_total = round(sum((float(i.get('price', 0) or 0) * int(i.get('quantity', 1) or 1)) for i in user_cart), 2)
    return jsonify({'message': 'Cart updated', 'total': updated_total, 'items': len(user_cart)})


@app.route('/checkout', methods=['GET', 'POST'])
def checkout_page():
    if 'user' not in session:
        return redirect('/')

    user_key = get_user_key()
    cart_store = load_cart_store()
    cart_items = cart_store.get(user_key, [])

    if not cart_items:
        return redirect(url_for('view_cart'))

    estimated_total = round(
        sum(float(item.get('price', 0) or 0) * int(item.get('quantity', 1) or 1) for item in cart_items),
        2,
    )

    if request.method == 'POST':
        payment_method = request.form.get('payment_method', '').strip()
        transaction_ref = request.form.get('transaction_ref', '').strip()

        if payment_method not in {'UPI', 'Card', 'COD'}:
            return render_template(
                'checkout.html',
                cart_data=cart_items,
                estimated_total=estimated_total,
                error='Please select a valid payment method.'
            ), 400

        if payment_method in {'UPI', 'Card'} and not transaction_ref:
            return render_template(
                'checkout.html',
                cart_data=cart_items,
                estimated_total=estimated_total,
                error='Transaction reference is required for UPI/Card payment.'
            ), 400

        cart_store[user_key] = []
        save_cart_store(cart_store)

        success_message = f"Payment successful via {payment_method}."
        return render_template('checkout.html', cart_data=[], estimated_total=0, success=success_message)

    return render_template('checkout.html', cart_data=cart_items, estimated_total=estimated_total)


@app.route('/try_on', methods=['POST'])
def try_on():
    # Ensure the user is logged in
    if 'user' not in session:
        return jsonify({'error': 'User not logged in'}), 403

    # Get the uploaded clothing and avatar images
    clothing_image = request.files['clothing_image']
    avatar_image = request.files['avatar_image']

    # Save the uploaded files temporarily
    clothing_image_path = './uploads/clothing_image.jpg'
    avatar_image_path = './uploads/avatar_image.jpg'
    clothing_image.save(clothing_image_path)
    avatar_image.save(avatar_image_path)

    # Send the images to the third-party API
    headers = rapidapi_headers()
    if headers is None:
        return jsonify({'error': 'RAPIDAPI_KEY is not configured in environment'}), 500

    with open(clothing_image_path, 'rb') as clothing_file, open(avatar_image_path, 'rb') as avatar_file:
        files = {
            'clothing_image': clothing_file,
            'avatar_image': avatar_file
        }
        response = requests.post(RAPIDAPI_URL, headers=headers, files=files, timeout=(20, 180))
    
    # Process the response
    if response.status_code == 200:
        with open('./uploads/output_image.png', 'wb') as out_file:
            out_file.write(response.content)
        return jsonify({'image_url': url_for('static', filename='uploads/output_image.png')})
    else:
        return jsonify({'error': 'Error processing images'}), 500
    
@app.route('/upload', methods=['POST'])
def upload_file():
    clothing_temp_path = None
    avatar_temp_path = None
    try:
        # Get the uploaded files from the form
        clothing_image = request.files.get('clothing_image')
        avatar_image = request.files.get('avatar_image')

        # Ensure the files exist
        if not clothing_image or not avatar_image:
            return jsonify({"error": "Both clothing and avatar images are required"}), 400

        if clothing_image.filename == '' or avatar_image.filename == '':
            return jsonify({"error": "Please select valid image files"}), 400

        if not allowed_file(clothing_image.filename) or not allowed_file(avatar_image.filename):
            return jsonify({"error": "Only PNG, JPG, and JPEG files are allowed"}), 400

        # Print file details for debugging
        print(f"Clothing Image: {clothing_image.filename}, Size: {clothing_image.content_length}")
        print(f"Avatar Image: {avatar_image.filename}, Size: {avatar_image.content_length}")

        clothing_name = secure_filename(clothing_image.filename)
        avatar_name = secure_filename(avatar_image.filename)
        clothing_temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_clothing_{clothing_name}")
        avatar_temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f"temp_avatar_{avatar_name}")

        clothing_image.save(clothing_temp_path)
        avatar_image.save(avatar_temp_path)

        # Prepare files for the API request
        with open(clothing_temp_path, 'rb') as clothing_file, open(avatar_temp_path, 'rb') as avatar_file:
            files = {
                'clothing_image': (clothing_name, clothing_file, clothing_image.content_type or 'image/jpeg'),
                'avatar_image': (avatar_name, avatar_file, avatar_image.content_type or 'image/jpeg')
            }

            headers = rapidapi_headers()
            if headers is None:
                return jsonify({"error": "RAPIDAPI_KEY is not configured in environment"}), 500

            response = None
            timeout_attempts = [(20, 180), (25, 240)]
            last_exception = None
            for connect_timeout, read_timeout in timeout_attempts:
                try:
                    response = requests.post(
                        RAPIDAPI_URL,
                        files=files,
                        headers=headers,
                        timeout=(connect_timeout, read_timeout),
                    )
                    break
                except requests.Timeout as exc:
                    last_exception = exc
                    clothing_file.seek(0)
                    avatar_file.seek(0)

            if response is None and last_exception is not None:
                raise requests.Timeout(str(last_exception))

        # Handle the response from the API
        if response.status_code == 200:
            # Save the output image directly to the 'uploads' folder
            output_filename = f"output_{uuid.uuid4().hex}.png"
            output_image_path = os.path.join(app.config['UPLOAD_FOLDER'], output_filename)
            with open(output_image_path, 'wb') as out_file:
                out_file.write(response.content)

            # Return the URL of the output image (serve it from the 'uploads' folder)
            return jsonify({'image_url': f'/uploads/{output_filename}'})

        # If the API response is not successful, return a friendly error message
        friendly_message = "Unable to process images. Please try with clearer and larger photos."
        try:
            error_payload = response.json()
            payload_text = json.dumps(error_payload).lower()
        except ValueError:
            error_payload = {}
            payload_text = (response.text or '').lower()

        if 'image_too_small' in payload_text or 'too small' in payload_text:
            if 'avatar' in payload_text:
                friendly_message = "Avatar image is too small. Please upload a larger avatar photo."
            elif 'clothing' in payload_text or 'cloth' in payload_text:
                friendly_message = "Clothing image is too small. Please upload a larger clothing photo."
            else:
                friendly_message = "Image is too small. Please upload a larger image."
        elif 'face' in payload_text and 'detect' in payload_text:
            friendly_message = "Face not detected clearly. Please upload a clearer front-facing avatar image."
        elif 'invalid' in payload_text or 'format' in payload_text:
            friendly_message = "Invalid image format. Please upload JPG, JPEG, or PNG files."

        status_code = response.status_code if 400 <= response.status_code < 600 else 500
        return jsonify({"error": friendly_message}), status_code

    except requests.Timeout:
        return jsonify({"error": "Upload timed out while processing with try-on API. Please try again."}), 504
    except requests.RequestException as e:
        return jsonify({"error": f"Network error while contacting try-on API: {str(e)}"}), 502

    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        if clothing_temp_path and os.path.exists(clothing_temp_path):
            os.remove(clothing_temp_path)
        if avatar_temp_path and os.path.exists(avatar_temp_path):
            os.remove(avatar_temp_path)


@app.errorhandler(RequestEntityTooLarge)
def handle_file_too_large(e):
    return jsonify({"error": "File too large. Maximum allowed size is 16MB."}), 413


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.9, min_tracking_confidence=0.9)
mp_face_detection = mp.solutions.face_detection
face_detector = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.6)

def _load_first_model(candidate_names, require_proba=False):
    for model_name in candidate_names:
        model_path = os.path.join("models", model_name)
        if not os.path.exists(model_path):
            continue
        try:
            model = joblib.load(model_path)
            if require_proba and not hasattr(model, 'predict_proba'):
                continue
            return model
        except Exception as exc:
            print(f"Unable to load model {model_name}: {exc}")
    return None


def _load_scaler(candidate_names):
    for scaler_name in candidate_names:
        scaler_path = os.path.join("models", scaler_name)
        if not os.path.exists(scaler_path):
            continue
        try:
            return joblib.load(scaler_path)
        except Exception as exc:
            print(f"Unable to load scaler {scaler_name}: {exc}")
    return None


gender_svm_model = _load_first_model([
    "gender_svm_model.pkl",
    "svm_gender_model.pkl",
], require_proba=True)
gender_rf_model = _load_first_model([
    "gender_rf_model.pkl",
    "rf_gender_model.pkl",
    "gender_random_forest_model.pkl",
], require_proba=True)
gender_scaler = _load_scaler([
    "gender_scaler.pkl",
    "svm_gender_scaler.pkl",
])

size_svm_model = _load_first_model([
    "size_svm_model.pkl",
    "svm_model.pkl",
], require_proba=True)
size_rf_model = _load_first_model([
    "size_rf_model.pkl",
    "rf_size_model.pkl",
], require_proba=True)
size_scaler = _load_scaler([
    "size_scaler.pkl",
    "svm_scaler.pkl",
])

if size_rf_model is None:
    fallback_rf = _load_first_model(["rf_model_height.pkl", "rf_model_waist.pkl"], require_proba=True)
    size_rf_model = fallback_rf

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def euclidean_distance(point1, point2):
    return np.sqrt((point1.x - point2.x) ** 2 + (point1.y - point2.y) ** 2)


def detect_face(image_bgr):
    try:
        image_h, image_w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = face_detector.process(image_rgb)
        if not results.detections:
            return None, "No face detected"

        largest_area = -1
        selected_box = None
        for detection in results.detections:
            box = detection.location_data.relative_bounding_box
            x = max(0, int(box.xmin * image_w))
            y = max(0, int(box.ymin * image_h))
            width = int(box.width * image_w)
            height = int(box.height * image_h)
            width = min(width, image_w - x)
            height = min(height, image_h - y)

            if width <= 0 or height <= 0:
                continue

            area = width * height
            if area > largest_area:
                largest_area = area
                selected_box = (x, y, width, height)

        if selected_box is None:
            return None, "No valid face bounding box detected"

        x, y, width, height = selected_box
        face_crop = image_bgr[y:y + height, x:x + width]
        if face_crop.size == 0:
            return None, "Face crop failed"

        return face_crop, None
    except Exception as exc:
        return None, f"Face detection error: {str(exc)}"


def _extract_class_probability(model, probabilities, positive_labels):
    classes = [str(cls).lower() for cls in getattr(model, 'classes_', [])]
    for label in positive_labels:
        if label in classes:
            return float(probabilities[classes.index(label)])
    if len(probabilities) == 2:
        return float(probabilities[1])
    return float(np.max(probabilities))


def _normalize_probability(value):
    try:
        numeric = float(value)
    except (TypeError, ValueError):
        return None

    if numeric > 1.0:
        numeric = numeric / 100.0
    return float(np.clip(numeric, 0.0, 1.0))


def _build_gender_result(male_probability, warning=None):
    male_probability = _normalize_probability(male_probability)
    if male_probability is None:
        male_probability = 0.5

    female_probability = float(np.clip(1.0 - male_probability, 0.0, 1.0))
    confidence = round(max(male_probability, female_probability) * 100, 2)

    predicted_gender = 'male' if male_probability >= female_probability else 'female'

    result = {
        "predicted_gender": predicted_gender,
        "confidence": confidence,
        "probabilities": {
            "male": round(male_probability * 100, 2),
            "female": round(female_probability * 100, 2),
        }
    }
    if warning:
        result["warning"] = warning
    return result


def _gender_heuristic_probability(image_bgr):
    try:
        pose_features, _, pose_error = extract_pose_features(image_bgr, 'Uncertain')
        if pose_error or pose_features is None:
            return 0.5

        shoulder_width, hip_width, *_others = pose_features[0]
        shoulder_hip_ratio = shoulder_width / max(hip_width, 1e-6)

        if shoulder_hip_ratio >= 1.12:
            return 0.72
        if shoulder_hip_ratio <= 0.95:
            return 0.28
        return 0.5
    except Exception:
        return 0.5


def _deepface_gender_probability(face_crop_bgr):
    if not DEEPFACE_AVAILABLE:
        return None
    try:
        face_rgb = cv2.cvtColor(face_crop_bgr, cv2.COLOR_BGR2RGB)
        result = DeepFace.analyze(
            img_path=face_rgb,
            actions=['gender'],
            detector_backend='skip',
            enforce_detection=False,
            silent=True,
        )
        if isinstance(result, list):
            result = result[0] if result else {}
        gender_scores = result.get('gender', {}) or {}
        normalized_scores = {str(k).strip().lower(): v for k, v in gender_scores.items()}
        male = _normalize_probability(normalized_scores.get('man', normalized_scores.get('male', 0)))
        female = _normalize_probability(normalized_scores.get('woman', normalized_scores.get('female', 0)))

        if male is None:
            male = 0.0
        if female is None:
            female = 0.0

        score_sum = male + female
        if score_sum <= 0:
            return None
        return male / score_sum
    except Exception:
        return None


def _deepface_gender_probability_full(image_bgr):
    if not DEEPFACE_AVAILABLE:
        return None
    try:
        result = DeepFace.analyze(
            img_path=image_bgr,
            actions=['gender'],
            detector_backend='opencv',
            enforce_detection=False,
            silent=True,
        )
        if isinstance(result, list):
            result = result[0] if result else {}
        gender_scores = result.get('gender', {}) or {}
        normalized_scores = {str(k).strip().lower(): v for k, v in gender_scores.items()}
        male = _normalize_probability(normalized_scores.get('man', normalized_scores.get('male', 0)))
        female = _normalize_probability(normalized_scores.get('woman', normalized_scores.get('female', 0)))

        if male is None:
            male = 0.0
        if female is None:
            female = 0.0

        score_sum = male + female
        if score_sum <= 0:
            return None
        return male / score_sum
    except Exception:
        return None


def predict_gender(image_bgr):
    try:
        full_image_deepface_prob = _deepface_gender_probability_full(image_bgr)

        face_crop, error = detect_face(image_bgr)
        if error or face_crop is None:
            male_avg = _gender_heuristic_probability(image_bgr)
            if full_image_deepface_prob is not None:
                male_avg = (full_image_deepface_prob * 0.8) + (male_avg * 0.2)
            return _build_gender_result(male_avg, warning="Face not detected clearly; used fallback inference.")

        gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (64, 64))
        normalized = resized.astype(np.float32) / 255.0
        flat = normalized.flatten().reshape(1, -1)

        features_for_models = gender_scaler.transform(flat) if gender_scaler is not None else flat

        male_probabilities = []
        if full_image_deepface_prob is not None:
            male_probabilities.append(full_image_deepface_prob)

        deepface_prob = _deepface_gender_probability(face_crop)
        if deepface_prob is not None:
            male_probabilities.append(deepface_prob)

        if gender_svm_model is not None and hasattr(gender_svm_model, 'predict_proba'):
            svm_proba = gender_svm_model.predict_proba(features_for_models)[0]
            male_probabilities.append(
                _extract_class_probability(gender_svm_model, svm_proba, ['male', 'm', '1'])
            )

        if gender_rf_model is not None and hasattr(gender_rf_model, 'predict_proba'):
            rf_proba = gender_rf_model.predict_proba(features_for_models)[0]
            male_probabilities.append(
                _extract_class_probability(gender_rf_model, rf_proba, ['male', 'm', '1'])
            )

        if male_probabilities:
            male_avg = float(np.mean(male_probabilities))
        else:
            male_avg = _gender_heuristic_probability(image_bgr)

        model_disagreement = abs((2.0 * male_avg) - 1.0)
        if model_disagreement < 0.1:
            if deepface_prob is not None:
                male_avg = (male_avg * 0.6) + (deepface_prob * 0.4)
            elif full_image_deepface_prob is not None:
                male_avg = (male_avg * 0.7) + (full_image_deepface_prob * 0.3)
            else:
                pose_male = _gender_heuristic_probability(image_bgr)
                male_avg = (male_avg * 0.85) + (pose_male * 0.15)

        return _build_gender_result(male_avg)
    except Exception as exc:
        male_avg = _gender_heuristic_probability(image_bgr)
        model_disagreement = abs((2.0 * male_avg) - 1.0)
        if model_disagreement < 0.08:
            pose_male = _gender_heuristic_probability(image_bgr)
            male_avg = (male_avg * 0.7) + (pose_male * 0.3)
        return _build_gender_result(male_avg, warning=f"Gender fallback used: {str(exc)}")

def calculate_measurements(landmarks, image_height, image_width):
    # Calculate body measurements using MediaPipe landmarks
    measurements = {}
    
    # Arm measurements
    measurements['left_arm'] = euclidean_distance(
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value]
    ) * image_height

    measurements['right_arm'] = euclidean_distance(
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value]
    ) * image_height

    # Leg measurements
    measurements['left_leg'] = euclidean_distance(
        landmarks[mp_pose.PoseLandmark.LEFT_HIP.value],
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value]
    ) * image_height

    measurements['right_leg'] = euclidean_distance(
        landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value],
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value]
    ) * image_height

    # Shoulder width
    measurements['shoulder_width'] = abs(
        landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x -
        landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x
    ) * image_width

    # Estimate height using full body landmarks
    top_point = min(
        landmarks[mp_pose.PoseLandmark.NOSE.value].y,
        landmarks[mp_pose.PoseLandmark.LEFT_EAR.value].y,
        landmarks[mp_pose.PoseLandmark.RIGHT_EAR.value].y
    )
    bottom_point = max(
        landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y,
        landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y
    )
    measurements['estimated_height'] = (bottom_point - top_point) * image_height

    # Estimate waist using hip points
    left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
    right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
    measurements['estimated_waist'] = euclidean_distance(left_hip, right_hip) * image_width * 2

    return measurements

def extract_pose_features(image_bgr, gender_label='Uncertain'):
    try:
        image_h, image_w = image_bgr.shape[:2]
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        results = pose.process(image_rgb)
        if not results.pose_landmarks:
            return None, None, "No pose landmarks detected"

        landmarks = results.pose_landmarks.landmark

        def to_px(index):
            lm = landmarks[index]
            return np.array([lm.x * image_w, lm.y * image_h], dtype=np.float32)

        left_shoulder = to_px(11)
        right_shoulder = to_px(12)
        left_hip = to_px(23)
        right_hip = to_px(24)
        nose = to_px(0)
        left_ankle = to_px(27)
        right_ankle = to_px(28)
        left_wrist = to_px(15)
        right_wrist = to_px(16)

        mid_hip = (left_hip + right_hip) / 2.0
        mid_ankle = (left_ankle + right_ankle) / 2.0

        shoulder_width = np.linalg.norm(left_shoulder - right_shoulder) / max(image_h, 1)
        hip_width = np.linalg.norm(left_hip - right_hip) / max(image_h, 1)
        torso_height = np.linalg.norm(nose - mid_hip) / max(image_h, 1)
        full_body_height = np.linalg.norm(nose - mid_ankle) / max(image_h, 1)

        shoulder_width_px = np.linalg.norm(left_shoulder - right_shoulder)
        hip_width_px = np.linalg.norm(left_hip - right_hip)
        torso_height_px = np.linalg.norm(nose - mid_hip)
        full_body_height_px = np.linalg.norm(nose - mid_ankle)
        left_arm_px = np.linalg.norm(left_shoulder - left_wrist)
        right_arm_px = np.linalg.norm(right_shoulder - right_wrist)
        avg_arm_px = (left_arm_px + right_arm_px) / 2.0

        left_shoulder_vis = landmarks[11].visibility
        right_shoulder_vis = landmarks[12].visibility
        left_hip_vis = landmarks[23].visibility
        right_hip_vis = landmarks[24].visibility
        nose_vis = landmarks[0].visibility
        left_ankle_vis = landmarks[27].visibility
        right_ankle_vis = landmarks[28].visibility

        ankle_visible = (left_ankle_vis >= 0.55 and right_ankle_vis >= 0.55)
        nose_visible = nose_vis >= 0.55
        full_body_visible = bool(ankle_visible and nose_visible)

        shoulder_ref_cm = 42.0
        hip_ref_cm = 36.0
        torso_ref_cm = 55.0
        if gender_label == 'male':
            shoulder_ref_cm = 44.0
            hip_ref_cm = 35.0
            torso_ref_cm = 58.0
        elif gender_label == 'female':
            shoulder_ref_cm = 40.0
            hip_ref_cm = 37.0
            torso_ref_cm = 54.0

        shoulder_scale = shoulder_ref_cm / max(shoulder_width_px, 1e-6)
        hip_scale = hip_ref_cm / max(hip_width_px, 1e-6)
        torso_scale = torso_ref_cm / max(torso_height_px, 1e-6)

        shoulder_conf = max(min((left_shoulder_vis + right_shoulder_vis) / 2.0, 1.0), 0.2)
        hip_conf = max(min((left_hip_vis + right_hip_vis) / 2.0, 1.0), 0.2)
        torso_conf = max(min((nose_vis + left_hip_vis + right_hip_vis) / 3.0, 1.0), 0.2)

        weighted_scale_sum = (shoulder_scale * shoulder_conf) + (hip_scale * hip_conf) + (torso_scale * torso_conf)
        weighted_conf_sum = shoulder_conf + hip_conf + torso_conf
        cm_per_px = weighted_scale_sum / max(weighted_conf_sum, 1e-6)

        shoulder_width_cm = shoulder_width_px * cm_per_px
        hip_bone_width_cm = hip_width_px * cm_per_px
        torso_length_cm = torso_height_px * cm_per_px
        arm_length_cm = avg_arm_px * cm_per_px

        if full_body_visible:
            estimated_height_cm = full_body_height_px * cm_per_px * 1.03
        else:
            estimated_height_cm = torso_length_cm * 3.04

        estimated_height_cm = float(np.clip(estimated_height_cm, 140.0, 210.0))

        if gender_label == 'male':
            chest_factor = 2.18
            waist_to_hip = 0.90
        elif gender_label == 'female':
            chest_factor = 2.06
            waist_to_hip = 0.84
        else:
            chest_factor = 2.12
            waist_to_hip = 0.87

        chest_width_cm = shoulder_width_cm * chest_factor
        hip_width_cm = hip_bone_width_cm * 2.12
        waist_width_cm = hip_width_cm * waist_to_hip

        chest_width_cm = float(np.clip(chest_width_cm, 70.0, 150.0))
        waist_width_cm = float(np.clip(waist_width_cm, 55.0, 140.0))
        hip_width_cm = float(np.clip(hip_width_cm, 70.0, 155.0))

        shoulder_hip_ratio = shoulder_width / max(hip_width, 1e-6)
        shoulder_torso_ratio = shoulder_width / max(torso_height, 1e-6)
        hip_torso_ratio = hip_width / max(torso_height, 1e-6)

        features = np.array([
            shoulder_width,
            hip_width,
            torso_height,
            full_body_height,
            shoulder_hip_ratio,
            shoulder_torso_ratio,
            hip_torso_ratio,
        ], dtype=np.float32).reshape(1, -1)

        print("Pose Features:", features.tolist()[0])

        feature_payload = {
            "arm_length_cm": round(float(arm_length_cm), 2),
            "shoulder_width_cm": round(float(shoulder_width_cm), 2),
            "chest_width_cm": round(float(chest_width_cm), 2),
            "waist_width_cm": round(float(waist_width_cm), 2),
            "hip_width_cm": round(float(hip_width_cm), 2),
            "torso_length_cm": round(float(torso_length_cm), 2),
            "estimated_height_cm": round(float(estimated_height_cm), 2) if estimated_height_cm is not None else None,
            "calibration_quality": round(float((weighted_conf_sum / 3.0) * 100.0), 2),
            "shoulder_width": round(float(shoulder_width), 5),
            "hip_width": round(float(hip_width), 5),
            "torso_height": round(float(torso_height), 5),
            "full_body_height": round(float(full_body_height), 5),
            "shoulder_to_hip_ratio": round(float(shoulder_hip_ratio), 5),
            "shoulder_to_torso_ratio": round(float(shoulder_torso_ratio), 5),
            "hip_to_torso_ratio": round(float(hip_torso_ratio), 5),
        }
        return features, feature_payload, None
    except Exception as exc:
        return None, None, f"Pose feature extraction error: {str(exc)}"


def build_fallback_pose_features(image_bgr, gender_label='Uncertain'):
    image_h, image_w = image_bgr.shape[:2]
    aspect_ratio = float(image_h) / max(float(image_w), 1.0)

    shoulder_width = min(max(0.18, 0.23 + (aspect_ratio - 1.4) * 0.02), 0.28)
    hip_width = min(max(0.16, shoulder_width * 0.9), 0.26)
    torso_height = min(max(0.26, 0.34 + (aspect_ratio - 1.4) * 0.03), 0.48)
    full_body_height = min(max(0.65, 0.82 + (aspect_ratio - 1.4) * 0.06), 1.05)

    shoulder_ref_cm = 42.0
    fallback_height_cm = 171.0
    waist_to_hip = 0.87
    chest_factor = 2.12
    if gender_label == 'male':
        shoulder_ref_cm = 44.0
        fallback_height_cm = 176.0
        waist_to_hip = 0.90
        chest_factor = 2.18
    elif gender_label == 'female':
        shoulder_ref_cm = 40.0
        fallback_height_cm = 163.0
        waist_to_hip = 0.84
        chest_factor = 2.06

    shoulder_width_cm = shoulder_ref_cm
    chest_width_cm = shoulder_width_cm * chest_factor
    hip_width_cm = shoulder_width_cm * 2.0
    waist_width_cm = hip_width_cm * waist_to_hip
    torso_length_cm = shoulder_width_cm * 1.46
    arm_length_cm = shoulder_width_cm * 1.34
    estimated_height_cm = fallback_height_cm + ((aspect_ratio - 1.5) * 10.0)
    estimated_height_cm = float(np.clip(estimated_height_cm, 150.0, 205.0))

    shoulder_hip_ratio = shoulder_width / max(hip_width, 1e-6)
    shoulder_torso_ratio = shoulder_width / max(torso_height, 1e-6)
    hip_torso_ratio = hip_width / max(torso_height, 1e-6)

    features = np.array([
        shoulder_width,
        hip_width,
        torso_height,
        full_body_height,
        shoulder_hip_ratio,
        shoulder_torso_ratio,
        hip_torso_ratio,
    ], dtype=np.float32).reshape(1, -1)

    payload = {
        "arm_length_cm": round(float(arm_length_cm), 2),
        "shoulder_width_cm": round(float(shoulder_width_cm), 2),
        "chest_width_cm": round(float(chest_width_cm), 2),
        "waist_width_cm": round(float(waist_width_cm), 2),
        "hip_width_cm": round(float(hip_width_cm), 2),
        "torso_length_cm": round(float(torso_length_cm), 2),
        "estimated_height_cm": round(float(estimated_height_cm), 2),
        "calibration_quality": 40.0,
        "shoulder_width": round(float(shoulder_width), 5),
        "hip_width": round(float(hip_width), 5),
        "torso_height": round(float(torso_height), 5),
        "full_body_height": round(float(full_body_height), 5),
        "shoulder_to_hip_ratio": round(float(shoulder_hip_ratio), 5),
        "shoulder_to_torso_ratio": round(float(shoulder_torso_ratio), 5),
        "hip_to_torso_ratio": round(float(hip_torso_ratio), 5),
    }

    return features, payload


def _probabilities_by_label(model, proba_vector, class_labels):
    mapped = {str(label): 0.0 for label in class_labels}
    model_classes = [str(cls) for cls in getattr(model, 'classes_', [])]
    if not model_classes:
        for idx, label in enumerate(class_labels):
            if idx < len(proba_vector):
                mapped[str(label)] = float(proba_vector[idx])
        return mapped

    for idx, cls_name in enumerate(model_classes):
        if idx < len(proba_vector) and cls_name in mapped:
            mapped[cls_name] = float(proba_vector[idx])
    return mapped


def _normalize_size_label(label):
    text = str(label).strip().upper()
    if text in {'XS', 'S', 'M', 'L', 'XL'}:
        return text

    numeric_map = {
        '0': 'XS',
        '1': 'S',
        '2': 'M',
        '3': 'L',
        '4': 'XL'
    }
    return numeric_map.get(text, text)


def _heuristic_size_prediction(features):
    size_labels = ['XS', 'S', 'M', 'L', 'XL']
    shoulder_width, hip_width, torso_height, _full_body_height, shoulder_hip_ratio, shoulder_torso_ratio, hip_torso_ratio = features[0]

    score = (
        (shoulder_hip_ratio * 0.34) +
        (shoulder_torso_ratio * 0.28) +
        (hip_torso_ratio * 0.20) +
        ((shoulder_width * 10.0) * 0.10) +
        ((hip_width * 10.0) * 0.08)
    )

    if score < 0.78:
        predicted = 'XS'
    elif score < 0.92:
        predicted = 'S'
    elif score < 1.06:
        predicted = 'M'
    elif score < 1.20:
        predicted = 'L'
    else:
        predicted = 'XL'

    base_conf = 0.66
    confidence = round(base_conf * 100, 2)

    probabilities = {label: 0.0 for label in size_labels}
    idx = size_labels.index(predicted)
    probabilities[predicted] = 0.62
    if idx - 1 >= 0:
        probabilities[size_labels[idx - 1]] = 0.19
    if idx + 1 < len(size_labels):
        probabilities[size_labels[idx + 1]] = 0.19

    return {
        "predicted_size": predicted,
        "confidence": confidence,
        "probabilities": {k: round(v * 100, 2) for k, v in probabilities.items()},
        "source": "heuristic"
    }


def predict_size_ensemble(features):
    try:
        features_for_models = size_scaler.transform(features) if size_scaler is not None else features

        model_probability_maps = []

        if size_svm_model is not None and hasattr(size_svm_model, 'predict_proba'):
            svm_proba = size_svm_model.predict_proba(features_for_models)[0]
            svm_classes = [_normalize_size_label(cls) for cls in getattr(size_svm_model, 'classes_', [])]
            if not svm_classes:
                svm_classes = ['XS', 'S', 'M', 'L', 'XL']
            model_probability_maps.append((size_svm_model, svm_proba, svm_classes))

        if size_rf_model is not None and hasattr(size_rf_model, 'predict_proba'):
            rf_proba = size_rf_model.predict_proba(features_for_models)[0]
            rf_classes = [_normalize_size_label(cls) for cls in getattr(size_rf_model, 'classes_', [])]
            if not rf_classes:
                rf_classes = ['XS', 'S', 'M', 'L', 'XL']
            model_probability_maps.append((size_rf_model, rf_proba, rf_classes))

        if not model_probability_maps:
            return _heuristic_size_prediction(features)

        all_classes = []
        for _model, _proba, classes in model_probability_maps:
            for cls_name in classes:
                normalized = _normalize_size_label(cls_name)
                if normalized not in all_classes:
                    all_classes.append(normalized)
        if not all_classes:
            all_classes = ['XS', 'S', 'M', 'L', 'XL']

        ensemble_map = {cls: 0.0 for cls in all_classes}
        for model, proba, _classes in model_probability_maps:
            mapped = _probabilities_by_label(model, proba, all_classes)
            mapped_normalized = {_normalize_size_label(k): v for k, v in mapped.items()}
            for cls in all_classes:
                ensemble_map[cls] += mapped_normalized.get(cls, 0.0)

        model_count = len(model_probability_maps)
        for cls in all_classes:
            ensemble_map[cls] = ensemble_map[cls] / max(model_count, 1)

        final_size = max(ensemble_map, key=ensemble_map.get)
        confidence = round(ensemble_map[final_size] * 100, 2)
        return {
            "predicted_size": final_size,
            "confidence": confidence,
            "probabilities": {k: round(v * 100, 2) for k, v in ensemble_map.items()},
            "source": "ensemble"
        }
    except Exception as exc:
        try:
            fallback = _heuristic_size_prediction(features)
            fallback["warning"] = f"Size model inference failed: {str(exc)}"
            return fallback
        except Exception as fallback_exc:
            return {"error": f"Size prediction error: {str(fallback_exc)}"}

@app.route('/predict_size', methods=['POST'])
def predict_size():
    filepath = None
    try:
        # Check if file was uploaded
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400

        # Save and process the image
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"pose_{uuid.uuid4().hex}_{filename}")
        file.save(filepath)

        # Read image
        img = cv2.imread(filepath)
        if img is None:
            return jsonify({'error': 'Failed to read image'}), 400

        gender_result = predict_gender(img)

        predicted_gender = str(gender_result.get('predicted_gender', '')).strip().lower()
        if predicted_gender not in {'male', 'female'}:
            probs = gender_result.get('probabilities', {}) or {}
            male_prob = float(probs.get('male', 50.0) or 50.0)
            female_prob = float(probs.get('female', 50.0) or 50.0)
            predicted_gender = 'male' if male_prob >= female_prob else 'female'

        pose_features, feature_payload, pose_error = extract_pose_features(img, predicted_gender)
        if pose_error or pose_features is None or feature_payload is None:
            pose_features, feature_payload = build_fallback_pose_features(img, predicted_gender)

        size_result = predict_size_ensemble(pose_features)
        if size_result.get('error'):    
            size_result = _heuristic_size_prediction(pose_features)
            size_result['warning'] = size_result.get('warning') or 'Fallback size estimation used.'

        # Prepare response
        estimated_size = size_result.get('predicted_size')
        if not estimated_size:
            fallback_size = _heuristic_size_prediction(pose_features)
            estimated_size = fallback_size.get('predicted_size', 'M')
            size_result['confidence'] = fallback_size.get('confidence', size_result.get('confidence', 0))
            size_result['probabilities'] = fallback_size.get('probabilities', size_result.get('probabilities', {}))
            size_result['source'] = fallback_size.get('source', size_result.get('source', 'heuristic'))
            size_result['warning'] = size_result.get('warning') or 'Size fallback result used.'

        response = {
            'gender': predicted_gender,
            'predicted_gender': predicted_gender,
            'gender_confidence': float(gender_result.get('confidence', 0) or 0),
            'gender_probabilities': gender_result.get('probabilities', {'male': 50.0, 'female': 50.0}),
            'gender_warning': gender_result.get('warning'),
            'estimated_size': str(estimated_size),
            'size_confidence': float(size_result.get('confidence', 0) or 0),
            'size_probabilities': size_result.get('probabilities', {}),
            'size_source': size_result.get('source', 'heuristic'),
            'size_warning': size_result.get('warning'),
            'measurements': feature_payload or {},
        }

        return jsonify(response)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if filepath and os.path.exists(filepath):
            os.remove(filepath)


if __name__ == '__main__':
    app.run(debug=True)