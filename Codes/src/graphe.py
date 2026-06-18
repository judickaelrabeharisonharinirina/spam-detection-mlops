from Traitement import *
    
#vectorization
x  = data["mess_clean"]
y  = data["Category"]
vector = TfidfVectorizer(stop_words='english')
X_train, X_test, y_train, y_test = train_test_split( x, y, test_size=0.2, random_state=1)
features = vector.fit_transform(X_train)
#creer un graph visuel
mots = vector.get_feature_names_out()
scores_moyens = features.mean(axis=0).A1
visu = pd.DataFrame({
    'mot': mots,
    'score': scores_moyens
})
top20 = visu.sort_values(by='score', ascending=False).head(20)

#construction graph
#encodage des labels
encoder = LabelEncoder()
y_encoded = encoder.fit_transform(y_train)
#réduction de dimension
pca = PCA(n_components=2)
X_pca = pca.fit_transform(features.toarray())
#dataFrame pour visualisation
df_pca = pd.DataFrame({
    'PCA1': X_pca[:, 0],
    'PCA2': X_pca[:, 1],
    'Category': y_train.values
})
#scatter plot
plt.figure(figsize=(12,8))
sns.scatterplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    hue=y_train,
    palette=['blue','red'],
    s=60,
    alpha=0.8
)

#appercu graphique
plt.title("Projection PCA des messages TF-IDF")
plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.2f}% variance)")
plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.2f}% variance)")
plt.tight_layout()

#save 
nom_dossier = "Figure"
if not os.path.exists(nom_dossier):
    os.makedirs(nom_dossier)
chemin_enregistrement = os.path.join(nom_dossier, "scatter.png")
plt.savefig(chemin_enregistrement, dpi=300, bbox_inches='tight')
