import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(BASE_DIR, "../Raw/USvideos.csv")
df=pd.read_csv(csv_path)
coldrop=['video_id', 'thumbnail_link', 'description', 'channel_title']
df=df.drop(columns=coldrop)
df=df.dropna()
print(df.columns)



# 4. Dates ko Asli 'Date Format' mein convert karna
print("Formatting dates...")

# Publish Time ko date mein badalna (aur time/timezone hatana)
df['publish_time'] = pd.to_datetime(df['publish_time'])
df['publish_date'] = pd.to_datetime(df['publish_time'].dt.date) # Sirf date rakhi

# Trending Date ka format ajeeb hota hai (Year.Day.Month), isko theek karna
df['trending_date'] = pd.to_datetime(df['trending_date'], format='%y.%d.%m')

# 5. Time Difference nikalna
df['time_to_trend'] = df['trending_date'] - df['publish_date']

# 6. THE 24-HOUR FILTER (Magic Step!)
# Sirf un rows ko rakho jahan time difference 1 din (1 day) ya usse kam ho
df = df[df['time_to_trend'] <= pd.Timedelta(days=1)]

# 7. Metadata ko Numbers mein badalna (Feature Engineering)
print("Extracting features from metadata...")

# Title kitna lamba hai?
df['title_length'] = df['title'].str.len()

# Video mein kitne tags use hue hain? (Tags '|' se jude hote hain)
df['tag_count'] = df['tags'].str.count('\|') + 1

# Video kis ghante upload hui? (Timing bahut zaroori hoti hai)
df['upload_hour'] = df['publish_time'].dt.hour

# 8. Target (y) Set Karna: "Viral" ki definition
# Agar pehle 24 ghante mein 50,000 se zyada views hain, toh Viral (1), warna Normal (0)
df['is_viral'] = (df['views'] > 5000).astype(int)

X = df[['title_length', 'tag_count', 'upload_hour', 'category_id']]
y = df['is_viral']

X_train,X_test,y_train,y_test=train_test_split(X,y, test_size=0.10,random_state=42)
model=DecisionTreeClassifier()
model.fit(X_train,y_train)
# Testing: See how accurate it is
accuracy = model.score(X_test, y_test)
print(f"Training Complete! Model Accuracy: {accuracy * 100:.2f}%")

# Saving: Exporting to models/ folder
model_path = os.path.join(BASE_DIR, 'models/viral.pkl')
joblib.dump(model, model_path)
print("Model saved to models/viral_predictor.pkl")













