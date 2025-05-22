import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, cross_val_predict
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import cv2
import os
from PIL import Image
import warnings
warnings.filterwarnings('ignore')

class ORLFaceRecognizer:
    def __init__(self, n_components=50, kernel='rbf'):
        """
        Inicializa o reconhecedor facial com PCA e SVM
        
        Args:
            n_components (int): Número de componentes principais para PCA
            kernel (str): Kernel para SVM
        """
        self.n_components = n_components
        self.kernel = kernel
        self.pca = PCA(n_components=n_components)
        self.svm = SVC(kernel=kernel, probability=True, random_state=42)
        self.scaler = StandardScaler()
        self.images = []
        self.labels = []
        self.image_size = (112, 92)  # Tamanho padrão ORL
        
    def load_orl_database(self, dataset_path):
            """
            Carrega a base de dados ORL

            Args:
                dataset_path (str): Caminho para a pasta do dataset ORL
            """
            print("Carregando base de dados ORL...")

            if not os.path.exists(dataset_path):
                print("Dataset ORL não encontrado. Criando dados sintéticos para demonstração...")
                self._create_synthetic_data()
                return

            self.images = []
            self.labels = []

            for subject_id in range(1, 41):
                subject_path = os.path.join(dataset_path, f's{subject_id}')
                if os.path.exists(subject_path):
                    for img_id in range(1, 11):
                        # --- FIX: Define img_path here ---
                        img_path = os.path.join(subject_path, f'{img_id}.pgm') # Assuming .pgm extension
                        # --- End FIX ---
                        if os.path.exists(img_path):
                            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                            if img is not None:
                                img_resized = cv2.resize(img, self.image_size)
                                self.images.append(img_resized.flatten())
                                self.labels.append(subject_id - 1)  # Labels de 0 a 39

            if len(self.images) == 0:
                print("Nenhuma imagem foi carregada. Criando dados sintéticos...")
                self._create_synthetic_data()
            else:
                self.images = np.array(self.images)
                self.labels = np.array(self.labels)
                print(f"Carregadas {len(self.images)} imagens de {len(np.unique(self.labels))} sujeitos")
    
    def _create_synthetic_data(self):
        """
        Cria dados sintéticos para demonstração quando o dataset ORL não está disponível
        """
        print("Criando dados sintéticos (40 sujeitos, 10 imagens cada)...")
        np.random.seed(42)
        
        self.images = []
        self.labels = []
        
        for subject_id in range(40):  
            base_face = np.random.randn(self.image_size[0], self.image_size[1]) * 50 + 128
            
            for img_id in range(10):  
                noise = np.random.randn(self.image_size[0], self.image_size[1]) * 20
                rotation_noise = np.random.randn(self.image_size[0], self.image_size[1]) * 10
                
                synthetic_face = base_face + noise + rotation_noise
                synthetic_face = np.clip(synthetic_face, 0, 255)
                
                self.images.append(synthetic_face.flatten())
                self.labels.append(subject_id)
        
        self.images = np.array(self.images)
        self.labels = np.array(self.labels)
        print(f"Criados dados sintéticos: {len(self.images)} imagens de {len(np.unique(self.labels))} sujeitos")
    
    def preprocess_and_train(self):
        """
        Aplica PCA e treina o classificador SVM
        """
        print("Aplicando PCA para redução de dimensionalidade...")
        
        X_scaled = self.scaler.fit_transform(self.images)
        
        X_pca = self.pca.fit_transform(X_scaled)
        
        print(f"Dimensionalidade reduzida de {self.images.shape[1]} para {X_pca.shape[1]} componentes")
        print(f"Variância explicada pelos {self.n_components} componentes: {self.pca.explained_variance_ratio_.sum():.3f}")
        
        # Treina o SVM
        print("Treinando classificador SVM...")
        self.svm.fit(X_pca, self.labels)
        
        return X_pca
    
    def predict_image(self, test_image_path_or_array):
        """
        Classifica uma imagem de teste
        Args:
            test_image_path_or_array: Caminho para imagem ou array numpy
        
        Returns:
            Resultado da classificação
        """
        if isinstance(test_image_path_or_array, str):
            if os.path.exists(test_image_path_or_array):
                test_img = cv2.imread(test_image_path_or_array, cv2.IMREAD_GRAYSCALE)
                test_img = cv2.resize(test_img, self.image_size)
            else:
                print("Imagem de teste não encontrada. Usando imagem sintética...")
                test_img = self._create_test_image()
        else:
            test_img = test_image_path_or_array
        
        test_img_flat = test_img.flatten().reshape(1, -1)
        test_img_scaled = self.scaler.transform(test_img_flat)
        test_img_pca = self.pca.transform(test_img_scaled)
        
        predicted_class = self.svm.predict(test_img_pca)[0]
        probabilities = self.svm.predict_proba(test_img_pca)[0]
        
        top5_indices = np.argsort(probabilities)[-5:][::-1]
        top5_classes = top5_indices
        top5_probs = probabilities[top5_indices]
        
        return {
            'test_image': test_img,
            'predicted_class': predicted_class,
            'top5_classes': top5_classes,
            'top5_probabilities': top5_probs,
            'all_probabilities': probabilities
        }
    
    def _create_test_image(self):
        """
        Cria uma imagem de teste sintética baseada em um sujeito aleatório
        """
        test_subject = np.random.randint(0, 40)
        subject_images = self.images[self.labels == test_subject]
        
        if len(subject_images) > 0:
            # Pega uma imagem base e adiciona ruído
            base_img = subject_images[0].reshape(self.image_size)
            noise = np.random.randn(*self.image_size) * 15
            test_img = base_img + noise
            test_img = np.clip(test_img, 0, 255).astype(np.uint8)
            return test_img
        else:
            return np.random.randint(0, 256, self.image_size, dtype=np.uint8)
    
    def visualize_results(self, prediction_result):
        """
        Visualiza os resultados da classificação
        """
        fig, axes = plt.subplots(2, 5, figsize=(15, 6))
        fig.suptitle('Resultados do Reconhecimento Facial', fontsize=16)
        
        axes[0, 0].imshow(prediction_result['test_image'], cmap='gray')
        axes[0, 0].set_title(f'Teste\nClasse Prevista: {prediction_result["predicted_class"]}')
        axes[0, 0].axis('off')
        
        predicted_class = prediction_result['predicted_class']
        class_images = self.images[self.labels == predicted_class]
        
        for i in range(min(9, len(class_images))):
            row = i // 4
            col = (i % 4) + 1
            if row == 0:
                axes[row, col].imshow(class_images[i].reshape(self.image_size), cmap='gray')
                axes[row, col].set_title(f'Classe {predicted_class} - {i+1}')
                axes[row, col].axis('off')
            else:
                if col <= 4:  # Só mostra até 4 na segunda linha
                    axes[row, col-1].imshow(class_images[i].reshape(self.image_size), cmap='gray')
                    axes[row, col-1].set_title(f'Classe {predicted_class} - {i+1}')
                    axes[row, col-1].axis('off')
        
        for i in range(2):
            for j in range(5):
                if i == 0 and j == 0:
                    continue
                if i == 0 and j <= 4:
                    continue
                if i == 1 and j <= 3:
                    continue
                axes[i, j].axis('off')
        
        plt.tight_layout()
        plt.show()
        
        print("\n=== TOP-5 CLASSES MAIS PROVÁVEIS ===")
        for i, (class_id, prob) in enumerate(zip(prediction_result['top5_classes'], 
                                                prediction_result['top5_probabilities'])):
            print(f"{i+1}. Classe {class_id}: {prob:.4f} ({prob*100:.2f}%)")
    
    def cross_validation_analysis(self):
        """
        Realiza validação cruzada e gera matriz de confusão
        """
        print("\n=== ANÁLISE DE VALIDAÇÃO CRUZADA ===")
        
        # Prepara os dados
        X_scaled = self.scaler.transform(self.images)
        X_pca = self.pca.transform(X_scaled)
        
        cv_scores = cross_val_score(self.svm, X_pca, self.labels, cv=5, scoring='accuracy')
        print(f"Acurácia média (5-fold CV): {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
        print(f"Scores individuais: {cv_scores}")
        
        y_pred_cv = cross_val_predict(self.svm, X_pca, self.labels, cv=5)
        cm = confusion_matrix(self.labels, y_pred_cv)
        
        plt.figure(figsize=(12, 10))
        
        if len(np.unique(self.labels)) > 20:
            # Agrupa classes para visualização mais limpa
            cm_reduced = self._reduce_confusion_matrix(cm, 20)
            sns.heatmap(cm_reduced, annot=False, cmap='Blues', square=True)
            plt.title('Matriz de Confusão (5-fold CV) - Resumida (20x20)')
        else:
            sns.heatmap(cm, annot=True, cmap='Blues', square=True, fmt='d')
            plt.title('Matriz de Confusão (5-fold CV)')
        
        plt.xlabel('Classe Prevista')
        plt.ylabel('Classe Verdadeira')
        plt.show()
        
        return cv_scores, cm
    
    def _reduce_confusion_matrix(self, cm, target_size):
        """
        Reduz o tamanho da matriz de confusão para visualização
        """
        original_size = cm.shape[0]
        group_size = original_size // target_size
        
        reduced_cm = np.zeros((target_size, target_size))
        
        for i in range(target_size):
            for j in range(target_size):
                i_start, i_end = i * group_size, min((i + 1) * group_size, original_size)
                j_start, j_end = j * group_size, min((j + 1) * group_size, original_size)
                reduced_cm[i, j] = cm[i_start:i_end, j_start:j_end].sum()
        
        return reduced_cm
    
    def tsne_visualization(self):
        """
        Visualiza a projeção t-SNE dos vetores PCA
        """
        print("\n=== VISUALIZAÇÃO t-SNE ===")
        print("Calculando projeção t-SNE... (pode demorar alguns minutos)")
        
        # Prepara os dados PCA
        X_scaled = self.scaler.transform(self.images)
        X_pca = self.pca.transform(X_scaled)
        
        # Aplica t-SNE (usando uma amostra se o dataset for muito grande)
        n_samples = min(1000, len(X_pca))  # Limita para acelerar o processamento
        indices = np.random.choice(len(X_pca), n_samples, replace=False)
        X_sample = X_pca[indices]
        y_sample = self.labels[indices]
        
        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        X_tsne = tsne.fit_transform(X_sample)
        
        # Visualiza
        plt.figure(figsize=(12, 8))
        scatter = plt.scatter(X_tsne[:, 0], X_tsne[:, 1], c=y_sample, 
                            cmap='tab20', alpha=0.7, s=30)
        plt.colorbar(scatter, label='Classe')
        plt.title('Projeção t-SNE dos Vetores PCA\n(Colorido por Classe)')
        plt.xlabel('Componente t-SNE 1')
        plt.ylabel('Componente t-SNE 2')
        plt.grid(True, alpha=0.3)
        plt.show()
    
    def analyze_errors(self, prediction_result):
        """
        Analisa os erros do modelo
        """
        print("\n=== ANÁLISE DE ERROS E LIMITAÇÕES ===")
        
        # Análise geral
        X_scaled = self.scaler.transform(self.images)
        X_pca = self.pca.transform(X_scaled)
        y_pred = self.svm.predict(X_pca)
        
        accuracy = np.mean(y_pred == self.labels)
        print(f"Acurácia no conjunto de treinamento: {accuracy:.4f}")
        
        # Identifica classes com mais erros
        errors_per_class = {}
        for true_label, pred_label in zip(self.labels, y_pred):
            if true_label != pred_label:
                if true_label not in errors_per_class:
                    errors_per_class[true_label] = 0
                errors_per_class[true_label] += 1
        
        if errors_per_class:
            print("\nClasses com mais erros:")
            sorted_errors = sorted(errors_per_class.items(), key=lambda x: x[1], reverse=True)
            for class_id, error_count in sorted_errors[:5]:
                total_samples = np.sum(self.labels == class_id)
                error_rate = error_count / total_samples
                print(f"  Classe {class_id}: {error_count}/{total_samples} erros ({error_rate:.2%})")
        
        # Discussão das limitações
        print("\nLimitações observadas:")
        print("1. Sensibilidade à iluminação e pose")
        print("2. Dificuldade com expressões faciais muito diferentes")
        print("3. Confusão entre indivíduos com características similares")
        print("4. Dependência da qualidade da imagem de entrada")
        print("5. Número limitado de componentes PCA pode perder informações importantes")
        
        # Separabilidade
        print(f"\nSeparabilidade: {self.n_components} componentes PCA explicam "
              f"{self.pca.explained_variance_ratio_.sum():.1%} da variância total")


def main():
    """
    Função principal que executa todo o pipeline
    """
    print("=== SISTEMA DE RECONHECIMENTO FACIAL COM PCA + SVM ===")
    print("Base de dados: ORL (Olivetti Research Laboratory)")
    print("Método: PCA + SVM com kernel RBF\n")
    
    recognizer = ORLFaceRecognizer(n_components=50, kernel='rbf')
    
    dataset_path = "orl_faces"  
    recognizer.load_orl_database(dataset_path)
    
    X_pca = recognizer.preprocess_and_train()
    
    print("\n=== TESTE DE CLASSIFICAÇÃO ===")
    test_image_path = "test_image.pgm"  
    prediction_result = recognizer.predict_image(test_image_path)
    
    recognizer.visualize_results(prediction_result)
    
    cv_scores, cm = recognizer.cross_validation_analysis()
    
    recognizer.tsne_visualization()
    
    recognizer.analyze_errors(prediction_result)
    
    print("\n=== RESUMO DOS RESULTADOS ===")
    print(f"Acurácia média (5-fold CV): {cv_scores.mean():.4f}")
    print(f"Classe prevista para imagem de teste: {prediction_result['predicted_class']}")
    print(f"Confiança da predição: {prediction_result['top5_probabilities'][0]:.4f}")


if __name__ == "__main__":
    main()
