from pyspark.sql import SparkSession
from pyspark.sql.functions import col, regexp_replace, split, avg, count, round
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.sql.functions import min, max, avg, count

#start apache spark
print("Mulai Apache Spark...")
spark = SparkSession.builder.appName("AnalisisAmazonFixed").getOrCreate()
spark.sparkContext.setLogLevel("ERROR")

#baca data
print("Membaca Data dari HDFS...")
df = spark.read.csv("hdfs:_____//amazon.csv", header=True, inferSchema=True)

#data cleaning dan feature engineering
print("Membersihkan & Menambah Kolom Diskon...")

#bersihakan simbil dan tipe
df_clean = df.withColumn("actual_price_clean", 
                         regexp_replace(col("actual_price"), "[₹,]", "").cast("float")) \
             .withColumn("discounted_price_clean", 
                         regexp_replace(col("discounted_price"), "[₹,]", "").cast("float")) \
             .withColumn("rating_clean", 
                         col("rating").cast("float")) \
             .withColumn("main_category", 
                         split(col("category"), r"\|").getItem(0))

#hitung persen diskon: (Harga Asli - Harga Diskon) / Harga Asli
df_lengkap = df_clean.withColumn("discount_pct", 
                                 ((col("actual_price_clean") - col("discounted_price_clean")) / col("actual_price_clean")) * 100)

#filter ata valid
df_final = df_lengkap.filter(
    col("actual_price_clean").isNotNull() & 
    col("rating_clean").isNotNull() & 
    col("discount_pct").isNotNull()
)

#analisis statistik
print("Menghitung...")

hasil_analisis = df_final.groupby("main_category") \
                         .agg(
                             count("product_id").alias("total_produk"),
                             round(avg("rating_clean"), 2).alias("avg_rating"),
                             round(avg("discounted_price_clean"), 2).alias("avg_harga_jual"),
                             round(avg("discount_pct"), 1).alias("avg_diskon_persen") 
                         ) \
                         .orderBy("total_produk", ascending=False)

hasil_analisis.show()

print("Menhitung pengaruh antara harga dengan rating...")
# ==========================================
# 1. PROSES MODELING
# ==========================================
assembler = VectorAssembler(inputCols=["discounted_price_clean", "rating_clean"], outputCol="features")
data_vec = assembler.transform(df_final)

# Melatih Model
kmeans = KMeans().setK(3).setSeed(1)
model = kmeans.fit(data_vec)
predictions = model.transform(data_vec)

# ==========================================
# 2. EVALUASI MODEL
# ==========================================
evaluator = ClusteringEvaluator()
silhouette = evaluator.evaluate(predictions)
print(f"Silhouette Score: {silhouette}")

# ==========================================
# 3. CEK RENTANG HARGA PER CLUSTER (INI YANG BARU)
# ==========================================
print("--- Statistik Rentang Harga Per Cluster ---")
range_harga = predictions.groupBy("prediction") \
    .agg(
        min("discounted_price_clean").alias("Harga_Terendah"),
        max("discounted_price_clean").alias("Harga_Tertinggi"),
        avg("discounted_price_clean").alias("Rata_Rata_Harga"),
        count("prediction").alias("Jumlah_Produk")
    ) \
    .orderBy("prediction")

range_harga.show()

pdf = predictions.select("discount_pct", "main_category", "rating_clean", "prediction").toPandas()

# ==========================================
# BAGIAN 1: DISKON DENGAN RATING
# ==========================================
print("\n" + "="*40)
print("1. DISKON DENGAN RATING")
print("="*40)

print("\nJUMLAH ANGGOTA PER CLUSTER:")
predictions.groupBy("prediction").count().show()

print(f"EVALUASI MODEL (Silhouette Score): {silhouette}")

print("\nMEMBUAT GRAFIK YANG TEPAT (Diskon vs Rating)...")
plt.figure(figsize=(10, 6))
scatter1 = plt.scatter(
    pdf['discount_pct'], 
    pdf['rating_clean'], 
    c=pdf['prediction'], 
    cmap='viridis', alpha=0.6, edgecolors='w', s=60
)
plt.xlabel('Besaran Diskon (%)')
plt.ylabel('Rating Produk (1-5)')
plt.title('1.C Grafik: Hubungan Diskon dengan Rating (Warna Cluster)')
plt.grid(True, linestyle='--', alpha=0.3)
legend1 = plt.legend(*scatter1.legend_elements(), loc="lower left", title="Cluster")
plt.gca().add_artist(legend1)
plt.tight_layout()
plt.savefig('grafik_1_diskon_vs_rating.png')
print("   -> Grafik tersimpan: grafik_1_diskon_vs_rating.png")


print("\nSELESAI.")

spark.stop()
