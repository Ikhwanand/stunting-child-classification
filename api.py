from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List
import uvicorn
from agent import stunting_recommendation
import os

# Inisialisasi FastAPI
app = FastAPI(
    title="Stunting Classification API",
    description="API untuk klasifikasi status stunting pada balita",
    version="1.0.0"
)

# Load model dan encoder
try:
    # Load Random Forest model
    rf_model = joblib.load('./models/rf_models.joblib')
    
    # Load encoders
    le_gender = joblib.load('./models/le_gender.pkl')

    
    le_status = joblib.load('./models/le_status.pkl')
    
    # Load scaler
    scaler = joblib.load('./models/scaler.pkl')
    
    print("Model dan encoder berhasil dimuat!")
    
except FileNotFoundError as e:
    print(f"Error loading model: {e}")
    rf_model = None
    le_gender = None
    le_status = None
    scaler = None

# Pydantic models untuk request dan response
class ChildData(BaseModel):
    umur_bulan: int = Field(..., ge=0, le=60, description="Umur anak dalam bulan (0-60)")
    jenis_kelamin: str = Field(..., description="Jenis kelamin: 'laki-laki' atau 'perempuan'")
    tinggi_badan: float = Field(..., ge=40, le=120, description="Tinggi badan dalam cm (40-120)")
    
    class Config:
        schema_extra = {
            "example": {
                "umur_bulan": 24,
                "jenis_kelamin": "laki-laki",
                "tinggi_badan": 85.5
            }
        }

class PredictionResponse(BaseModel):
    status_stunting: str
    probabilitas: Dict[str, float]
    rekomendasi: str
    input_data: Dict[str, Any]

class BatchChildData(BaseModel):
    data: List[ChildData]

# Fungsi untuk memberikan rekomendasi
def get_recommendation(status: str, umur: int, tinggi: float, jenis_kelamin: str) -> str:
    recommendations = {
        "normal": "Pertahankan pola makan sehat dan gizi seimbang. Lakukan pemeriksaan rutin untuk memantau pertumbuhan.",
        "stunted": "Konsultasikan dengan dokter anak. Tingkatkan asupan protein, vitamin, dan mineral. Berikan makanan bergizi tinggi secara teratur.",
        "severely stunted": "SEGERA konsultasikan dengan dokter anak atau ahli gizi. Diperlukan intervensi gizi intensif dan pemantauan medis ketat.",
        "tinggi": "Anak memiliki tinggi di atas rata-rata. Pastikan pertumbuhan seimbang dengan berat badan yang sehat."
    }
    
    # Karena kita akan mengimplementasikan AI agent untuk menggenerate rekomendasi treatment berdasarkan penyakit, jadi kita membuat generate treatment baru
    if status == 'stunted':
        recommendations['stunted'] = stunting_recommendation(age=umur, height_cm=tinggi, gender=jenis_kelamin, country='Indonesia', disease=status)
    
    if status == "severely stunted":
        recommendations['severely stunted'] = stunting_recommendation(age=umur, height_cm=tinggi, gender=jenis_kelamin, country='Indonesia', disease=status)

    
    base_rec = recommendations.get(status, "Konsultasikan dengan tenaga kesehatan.")
    
    # Tambahan rekomendasi berdasarkan umur
    # if umur < 24:
    #     base_rec += " Fokus pada ASI eksklusif dan MPASI yang tepat."
    # else:
    #     base_rec += " Berikan makanan keluarga yang bergizi dan seimbang."
    
    return base_rec

# Endpoint untuk health check
@app.get("/")
async def root():
    return {
        "message": "Stunting Classification API",
        "status": "active",
        "model_loaded": rf_model is not None
    }

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "model_status": "loaded" if rf_model is not None else "not_loaded"
    }

# Endpoint untuk prediksi tunggal
@app.post("/predict", response_model=PredictionResponse)
async def predict_stunting(child_data: ChildData):
    if rf_model is None:
        raise HTTPException(status_code=500, detail="Model belum dimuat")
    
    try:
        # Validasi jenis kelamin
        if child_data.jenis_kelamin.lower() not in ['laki-laki', 'perempuan']:
            raise HTTPException(
                status_code=400, 
                detail="Jenis kelamin harus 'laki-laki' atau 'perempuan'"
            )
        
        # Encode jenis kelamin
        gender_encoded = le_gender.transform([child_data.jenis_kelamin.lower()])[0]
        
        # Prepare features
        features = np.array([[
            child_data.umur_bulan,
            gender_encoded,
            child_data.tinggi_badan
        ]])
        
        # Scale features
        features_scaled = scaler.transform(features)
        
        # Prediksi
        prediction = rf_model.predict(features_scaled)[0]
        probabilities = rf_model.predict_proba(features_scaled)[0]
        
        # Decode prediction
        status_stunting = le_status.inverse_transform([prediction])[0]
        
        # Buat dictionary probabilitas
        prob_dict = {}
        for i, class_name in enumerate(le_status.classes_):
            prob_dict[class_name] = float(probabilities[i])
        
        # Dapatkan rekomendasi
        gender_in_english = "boy" if child_data.jenis_kelamin.lower() == "laki-laki" else "girl"
        rekomendasi = get_recommendation(status=status_stunting, umur=child_data.umur_bulan, tinggi=child_data.tinggi_badan, jenis_kelamin=gender_in_english)

        
        return PredictionResponse(
            status_stunting=status_stunting,
            probabilitas=prob_dict,
            rekomendasi=rekomendasi,
            input_data={
                "umur_bulan": child_data.umur_bulan,
                "jenis_kelamin": child_data.jenis_kelamin,
                "tinggi_badan": child_data.tinggi_badan
            }
        )
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error dalam prediksi: {str(e)}")

# Endpoint untuk prediksi batch
@app.post("/predict_batch")
async def predict_batch(batch_data: BatchChildData):
    if rf_model is None:
        raise HTTPException(status_code=500, detail="Model belum dimuat")
    
    results = []
    
    for i, child_data in enumerate(batch_data.data):
        try:
            # Validasi jenis kelamin
            if child_data.jenis_kelamin.lower() not in ['laki-laki', 'perempuan']:
                results.append({
                    "index": i,
                    "error": "Jenis kelamin harus 'laki-laki' atau 'perempuan'"
                })
                continue
            
            # Encode jenis kelamin
            gender_encoded = le_gender.transform([child_data.jenis_kelamin.lower()])[0]
            
            # Prepare features
            features = np.array([[
                child_data.umur_bulan,
                gender_encoded,
                child_data.tinggi_badan
            ]])
            
            # Scale features
            features_scaled = scaler.transform(features)
            
            # Prediksi
            prediction = rf_model.predict(features_scaled)[0]
            probabilities = rf_model.predict_proba(features_scaled)[0]
            
            # Decode prediction
            status_stunting = le_status.inverse_transform([prediction])[0]
            
            # Buat dictionary probabilitas
            prob_dict = {}
            for j, class_name in enumerate(le_status.classes_):
                prob_dict[class_name] = float(probabilities[j])
            
            # Dapatkan rekomendasi
            gender_in_english = "boy" if child_data.jenis_kelamin.lower() == "laki-laki" else "girl"
            rekomendasi = get_recommendation(status=status_stunting, umur=child_data.umur_bulan, tinggi=child_data.tinggi_badan, jenis_kelamin=gender_in_english)
            
            results.append({
                "index": i,
                "status_stunting": status_stunting,
                "probabilitas": prob_dict,
                "rekomendasi": rekomendasi,
                "input_data": {
                    "umur_bulan": child_data.umur_bulan,
                    "jenis_kelamin": child_data.jenis_kelamin,
                    "tinggi_badan": child_data.tinggi_badan
                }
            })
            
        except Exception as e:
            results.append({
                "index": i,
                "error": f"Error dalam prediksi: {str(e)}"
            })
    
    return {"results": results}

# Endpoint untuk informasi model
@app.get("/model_info")
async def get_model_info():
    if rf_model is None:
        raise HTTPException(status_code=500, detail="Model belum dimuat")
    
    return {
        "model_type": "Random Forest Classifier",
        "n_estimators": rf_model.n_estimators,
        "features": ["Umur (bulan)", "Jenis Kelamin", "Tinggi Badan (cm)"],
        "classes": le_status.classes_.tolist(),
        "feature_importances": rf_model.feature_importances_.tolist()
    }

# Endpoint untuk statistik dataset (jika ada)
@app.get("/dataset_stats")
async def get_dataset_stats():
    try:
        # Coba load dataset jika ada
        df = pd.read_csv('data_balita.csv')
        
        stats = {
            "total_samples": len(df),
            "features": list(df.columns),
            "status_distribution": df['Status Stunting'].value_counts().to_dict(),
            "gender_distribution": df['Jenis Kelamin'].value_counts().to_dict(),
            "age_stats": {
                "mean": float(df['Umur (bulan)'].mean()),
                "min": float(df['Umur (bulan)'].min()),
                "max": float(df['Umur (bulan)'].max()),
                "std": float(df['Umur (bulan)'].std())
            },
            "height_stats": {
                "mean": float(df['Tinggi Badan (cm)'].mean()),
                "min": float(df['Tinggi Badan (cm)'].min()),
                "max": float(df['Tinggi Badan (cm)'].max()),
                "std": float(df['Tinggi Badan (cm)'].std())
            }
        }
        
        return stats
        
    except FileNotFoundError:
        raise HTTPException(status_code=404, detail="Dataset tidak ditemukan")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error membaca dataset: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080)