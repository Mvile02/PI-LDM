import os
import sys
import numpy as np

def main():
    if len(sys.argv) < 3:
        print("Usage: python extract_cluster.py [archivo.npz] [ID_del_cluster]")
        print("Ejemplo: python extract_cluster.py LSZH_2019_R14_kinematic_200pts_clust5.npz 2")
        return

    npz_name = sys.argv[1]
    cluster_id = int(sys.argv[2])
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    # Try finding the file in processed or clusters directory
    file_path = os.path.join(base_dir, "data", "clusters", npz_name)
    if not os.path.exists(file_path):
        file_path = os.path.join(base_dir, "data", "processed", npz_name)
        
    if not os.path.exists(file_path):
        print(f"Error: {file_path} no encontrado.")
        return

    print(f"Cargando dataset agrupado: {file_path}")
    data = np.load(file_path, allow_pickle=True)
    X = data['X'].astype(np.float32)
    y = data['y']
    
    # Filter by cluster target
    mask = (y == cluster_id)
    X_target = X[mask]
    
    if len(X_target) == 0:
        print(f"Error: No se ha encontrado ninguna trayectoria para el clúster {cluster_id}.")
        return
        
    print(f"El clúster {cluster_id} tiene {len(X_target)} maniobras únicas.")
    
    # Salvar el array resultante como un npy limpio compatible con PI-LDM
    output_filename = npz_name.replace(".npz", f"_C{cluster_id}.npy")
    output_path = os.path.join(base_dir, "data", "processed", output_filename)
    
    np.save(output_path, X_target)
    print(f"¡Éxito! Maniobras extraídas y guardadas en: {output_path}")
    print(f"-> Para entrenar tu modelo, configura en train.py: FILE_BASE = '{output_filename.replace('.npy','')}'")

if __name__ == "__main__":
    main()
