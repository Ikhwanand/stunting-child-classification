# test_api.py
import requests
import json 


# Base URL API
BASE_URL = "http://localhost:8080"



def test_single_prediction():
    """Test prediksi tunggal"""
    url = f"{BASE_URL}/predict"
    
    data = {
        "umur_bulan": 24,
        "jenis_kelamin": "laki-laki",
        "tinggi_badan": 85.0
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print("=== Hasil Prediksi Tunggal ===")
        print(f"Status Stunting: {result['status_stunting']}")
        print(f"Probabilitas: {result['probabilitas']}")
        print(f"Rekomendasi: {result['rekomendasi']}")
        print()
    else: 
        print(f"Error: {response.status_code} - {response.text}")
    


def test_batch_prediction():
    """Test prediksi batch"""
    url = f"{BASE_URL}/predict_batch"
    
    data = {
        "data": [
            {
                "umur_bulan": 12,
                "jenis_kelamin": "perempuan",
                "tinggi_badan": 70.0
            },
            {
                "umur_bulan": 36,
                "jenis_kelamin": "laki-laki",
                "tinggi_badan": 95.0
            },
            {
                "umur_bulan": 18,
                "jenis_kelamin": "perempuan",
                "tinggi_badan": 75.5
            }
        ]
    }
    
    response = requests.post(url, json=data)
    
    if response.status_code == 200:
        result = response.json()
        print("=== Hasil Prediksi Batch ===")
        for i, res in enumerate(result['results']):
            if 'error' not in res:
                print(f"Data {i+1}:")
                print(f"    Status: {res['status_stunting']}")
                print(f"    Probabilitas tertinggi: {max(res['probabilitas'], key=res['probabilitas'].get)}")
                print()
            else: 
                print(f"Data {i+1}: Error - {res['error']}")
    else:
        print(f"Error: {response.status_code} - {response.text}")
        



def test_model_info():
    """Test informasi model"""
    url = f"{BASE_URL}/model_info"
    
    response = requests.get(url)
    
    if response.status_code == 200:
        result = response.json()
        print("=== Informasi Model ===")
        print(f"Tipe Modle: {result['model_type']}")
        print(f"Jumlah Estimator: {result['n_estimators']}")
        print(f"Features: {result['features']}")
        print(f"Classes: {result['classes']}")
        print(f"Feature Importances: {result['feature_importances']}")
        print()
    else: 
        print(f"Error: {response.status_code} - {response.text}")
    





if __name__=='__main__':
    print("Testing stunting classification API")
    print("=" * 50)
    
    # Test health check
    response = requests.get(f"{BASE_URL}/health")
    if response.status_code == 200:
        print("✅ API is healthy")
        print()
        
        # Run tests
        test_single_prediction()
        # test_batch_prediction()
        test_model_info()
    else: 
        print('❌ API is not responding')
        
        