import os

class Config:
    """
    Contenedor de parámetros de configuración de la red RBF y del dataset.
    Los valores se pueden cargar desde consola, GUI o diccionario.
    """
    def __init__(self):
        # Rutas
        self.raw_data_path = ""
        self.processed_data_path = ""
        self.dataset_name = ""
        # Mapeo de columnas (se puede ajustar dinámicamente después de cargar datos)
        self.input_columns = []
        self.target_column = ""
        # Hiperparámetros RBF
        self.n_centros = 0
        self.error_optimo = 0.0
        self.max_iteraciones = 0
        self.tipo_particion = "80-10-10"
        self.random_state = None   # None = aleatorio, un entero = reproducible
        
        # Estos campos se llenarán al cargar el dataset
        self.n_entradas = 0
        self.n_salidas = 0
        self.n_patrones = 0

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """
        Crea Config a partir de un diccionario plano.
        Usado por: GUI, API REST, archivos YAML/JSON de configuración.
        Claves requeridas:
            dataset_name, input_columns (list), target_column,
            n_centros, error_optimo, max_iteraciones
        Claves opcionales (tienen default):
            raw_data_path (default: 'data/raw')
            processed_data_path (default: 'data/processed')
            tipo_particion (default: '80-10-10')
            random_state (default: None)
        """
        config = cls()
        config.raw_data_path       = d.get("raw_data_path" or "data/raw")
        config.processed_data_path = d.get("processed_data_path") or "data/processed"
        config.dataset_name        = d["dataset_name"]
        config.input_columns       = list(d["input_columns"])
        config.target_column       = d["target_column"]
        config.n_centros           = int(d["n_centros"])
        config.error_optimo        = float(d["error_optimo"])
        config.max_iteraciones     = int(d["max_iteraciones"])
        config.tipo_particion      = d.get("tipo_particion", "80-10-10")
        config.random_state        = d.get("random_state", None)

        os.makedirs(config.processed_data_path, exist_ok=True)
        return config
    
    @classmethod
    def from_console(cls):
        """
        Crea una instancia de Config preguntando al usuario por consola
        Realiza validaciones y vuelve a preguntar en caso de error.
        """
        config = cls()
        print("=== Configuración del sistema RBF ===")
        
        # Rutas
        while True:
            raw = input("Ruta de la carpeta de datos raw (default: data/raw): ").strip()
            if not raw:
                raw = "data/raw"
            if os.path.isdir(raw):
                config.raw_data_path = raw
                break
            else:
                print(f"Error: La carpeta '{raw}' no existe. Inténtelo de nuevo.")
        
        processed_default = "data/processed"
        processed_input = input(f"Ruta de la carpeta de datos procesados (default: {processed_default}): ").strip()
        processed = processed_input if processed_input else processed_default
        os.makedirs(processed, exist_ok=True)
        
        # Nombre del dataset (archivo JSON sin extensión)
        dataset_name = ""
        while not dataset_name:
            dataset_name = input("Nombre del dataset (sin .json, ej: dataset_2_clases): ").strip()
            if not dataset_name:
                print("El nombre no puede estar vacío.")
        
        # Mapeo de columnas (básico, se puede refinar tras carga)
        input_columns: list[str] = []
        while not input_columns:
            s = input("Columnas de entrada separadas por coma (ej: x1,x2,x3,x4): ").strip()
            input_columns = [c.strip() for c in s.split(",") if c.strip()]
            if not input_columns:
                print("Debe haber al menos una columna de entrada.")
        
        target_column = ""
        while not target_column:
            target_column = input("Columna objetivo (clase/etiqueta): ").strip()
            if not target_column:
                print("La columna objetivo no puede estar vacía.")
        
        # Hiperparámetros RBF
        # Guardaremos la validación final en el pipeline, pero podemos pedir un entero positivo.
        n_centros = 0
        while n_centros < 1:
            try:
                n_centros = int(input("Número de centros radiales (neuronas ocultas): ").strip())
                if n_centros < 1:
                    print("Debe ser un entero positivo.")
            except ValueError:
                print("Ingresa un número entero.")

        error_optimo = -1.0
        while not (0.0 <= error_optimo <= 0.1):
            try:
                error_optimo = float(input("Error de aproximación óptimo (0.0 – 0.1): ").strip())
                if not (0.0 <= error_optimo <= 0.1):
                    print("El valor debe estar entre 0.0 y 0.1.")
            except ValueError:
                print("Ingresa un número decimal.")
 
        max_iteraciones = 0
        while max_iteraciones < 1:
            try:
                max_iteraciones = int(input("Máximo de iteraciones: ").strip())
                if max_iteraciones < 1:
                    print("Debe ser un entero positivo.")
            except ValueError:
                print("Ingresa un número entero.")
 
        particion_in = input("Esquema de partición (80-10-10 / 70-15-15, default: 80-10-10): ").strip()
        tipo_particion = particion_in if particion_in in ("80-10-10", "70-15-15") else "80-10-10"
        if particion_in and particion_in not in ("80-10-10", "70-15-15"):
            print("Valor no reconocido, usando 80-10-10.")
 
        fijar = input("¿Fijar semilla para reproducibilidad? (s/n, default: n): ").strip().lower()
        random_state = 42 if fijar == "s" else None
 
        print("\nConfiguración ingresada correctamente.\n")
 
        return cls.from_dict({
            "raw_data_path":       raw,
            "processed_data_path": processed,
            "dataset_name":        dataset_name,
            "input_columns":       input_columns,
            "target_column":       target_column,
            "n_centros":           n_centros,
            "error_optimo":        error_optimo,
            "max_iteraciones":     max_iteraciones,
            "tipo_particion":      tipo_particion,
            "random_state":        random_state,
        })
    
    # Serialización
    def to_dict(self) -> dict:
        """
        Convierte la configuración a diccionario plano.
        Útil para: guardar en JSON, enviar a la GUI, loguear.
        """
        return {
            "raw_data_path":       self.raw_data_path,
            "processed_data_path": self.processed_data_path,
            "dataset_name":        self.dataset_name,
            "input_columns":       self.input_columns,
            "target_column":       self.target_column,
            "n_centros":           self.n_centros,
            "error_optimo":        self.error_optimo,
            "max_iteraciones":     self.max_iteraciones,
            "tipo_particion":      self.tipo_particion,
            "random_state":        self.random_state,
        }
    
    # Validaciones
    def validate_post_data(self):
        """
        Validaciones que solo pueden hacerse después de cargar el dataset.
        Se llama desde el pipeline, no desde el constructor.
        """
        if self.n_entradas == 0:
            raise ValueError(
                "n_entradas no ha sido establecido. Carga los datos primero."
            )
        if self.n_centros < self.n_entradas:
            raise ValueError(
                f"n_centros ({self.n_centros}) debe ser >= n_entradas ({self.n_entradas})."
            )