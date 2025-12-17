from modules.preprocess import load_data
from modules.content_based import content_recommendation

def main():
    df = load_data()
    print("Sistema de recomendación de películas")

    for movie in df["title"]:
        print("-", movie)

    choice = input("\nEscribe una película que te guste: ")

    try:
        recs = content_recommendation(df, choice)
        print("\nRecomendaciones:")
        for r in recs:
            print("-", r)
    except:
        print("Película no encontrada")

if __name__ == "__main__":
    main()
