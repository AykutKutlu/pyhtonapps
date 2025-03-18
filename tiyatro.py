import streamlit as st
import pandas as pd

def main():
    st.title("Zaman Çizelgesi ve Görevler")
    
    # Zaman Çizelgesi Verisi
    schedule_data = {
        "Olay": [
            "Temaların Belirlenmesi", "Metinlerin okunup seçimlerin tamamlanması", "Metin havuzu oluşturulması", 
            "Mesh-up yapılması (first draft)", "Mesh-up yapılması (first draft-review)", "Mesh-up yapılması (first draft-ekiple okuma)",
            "Mesh-up yapılması (first draft-metnin genel hatlarının belli olması)", "Mesh-up yapılması (first draft-geri dönüt toplantısı)",
            "Mesh-up yapılması (second draft)", "Mesh-up yapılması (second draft-review)", "Mesh-up yapılması (second draft-ekiple okuma)",
            "Mesh-up yapılması (second draft)(kayıt metni)", "Mesh-up yapılması (second draft)(son metin)", "eğitim çalışmaları",
            "eğitim çalışmaları", "eğitim çalışmaları", "pasaj çalışması (ezber)"
        ],
        "Tarih": [
            "22.03.2025", "31.03.2025", "04.04.2025", "06.04.2025", "08.04.2025", "12.04.2025", "16.04.2025", "19.04.2025",
            "03.05.2025", "06.05.2025", "14.05.2025", "17.05.2025", "24.05.2025", "17.03.2025", "24.03.2025", "28.03.2025", "16.03.2025"
        ]
    }
    
    schedule_df = pd.DataFrame(schedule_data)
    st.subheader("Zaman Çizelgesi")
    st.dataframe(schedule_df)
    
    # Görevler Verisi
    task_data = {
        "Grup": ["Sanem, Eda", "Öykü, Gökçe", "Öykü, Gökçe", "Erlo, Öykü", "Erol"],
        "Görev": ["Godot", "Oyun Sonu", "Mutlu Günler", "Molloy", "Malone Ölüyor"]
    }
    
    task_df = pd.DataFrame(task_data)
    
    # Yeni veri seti oluşturma (Pivot Wider gibi)
    expanded_task_data = []
    for _, row in task_df.iterrows():
        people = [p.strip() for p in row["Grup"].split(",")]
        for person in people:
            expanded_task_data.append({"Kişi": person, "Görev": row["Görev"]})
    
    expanded_task_df = pd.DataFrame(expanded_task_data)
    
    st.subheader("Görevler")
    st.dataframe(expanded_task_df)
    
    # Kullanıcıya göre görev belirleme
    selected_person = st.selectbox("Kişi seçin", sorted(expanded_task_df["Kişi"].unique()))
    
    filtered_tasks = expanded_task_df[expanded_task_df["Kişi"] == selected_person]
    
    st.subheader(f"{selected_person} için görevler")
    st.write(filtered_tasks if not filtered_tasks.empty else "Bu kişi için görev bulunamadı.")
    
    # Notlar Bölümü
    st.subheader("Notlar")
    notes = st.text_area("Not ekleyin:", "")
    if st.button("Notu Kaydet"):
        st.write("Kaydedilen Not:", notes)

if __name__ == "__main__":
    main()
