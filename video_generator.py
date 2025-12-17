import cv2
import os
from tqdm import tqdm

# --- CONFIGURARE ---
IMAGE_FOLDER = 'frames_output'
VIDEO_NAME = 'SPX_Volatility_Evolution.mp4'
FPS = 24  # Frames Per Second (recomandat: 20-30 pentru fluiditate)


def generate_video():
    # 1. Verificare folder
    if not os.path.exists(IMAGE_FOLDER):
        print(f"Eroare: Folderul '{IMAGE_FOLDER}' nu exista.")
        return

    # 2. Colectare si Sortare Imagini
    # Este CRITIC sa le sortam corect (frame_00000, frame_00001, etc.)
    images = [img for img in os.listdir(IMAGE_FOLDER) if img.endswith(".png")]

    # Sortare naturala/alfanumerica
    images.sort()

    if not images:
        print("Nu am gasit imagini PNG in folder.")
        return

    print(f"Am gasit {len(images)} cadre. Incep generarea video la {FPS} FPS...")

    # 3. Citire primul cadru pentru a determina dimensiunile
    frame_path = os.path.join(IMAGE_FOLDER, images[0])
    frame = cv2.imread(frame_path)
    height, width, layers = frame.shape

    # 4. Initializare Video Writer
    # Codecul 'mp4v' este standard pentru .mp4 si compatibil cu majoritatea playerelor
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video = cv2.VideoWriter(VIDEO_NAME, fourcc, FPS, (width, height))

    # 5. Scriere Cadre
    try:
        for image in tqdm(images, desc="Scriere Video"):
            frame_path = os.path.join(IMAGE_FOLDER, image)
            frame = cv2.imread(frame_path)

            # Verificare de siguranta (daca o imagine e corupta)
            if frame is None:
                print(f"Atentie: Nu pot citi {image}. Sarim peste.")
                continue

            # Asigurare ca dimensiunile sunt identice (rareori o problema daca vin din acelasi script)
            if frame.shape[:2] != (height, width):
                frame = cv2.resize(frame, (width, height))

            video.write(frame)

    except KeyboardInterrupt:
        print("\nOprit de utilizator. Video-ul partial va fi salvat.")

    # 6. Finalizare
    video.release()
    cv2.destroyAllWindows()

    print(f"\nSucces! Video salvat ca: {VIDEO_NAME}")
    print(f"Durata estimata: {len(images) / FPS:.1f} secunde.")


if __name__ == "__main__":
    generate_video()