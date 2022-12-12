import numpy as np
from PIL import Image
import hyperparameters as hp
import json
import pickle
import os

def get_data(artists):
    for artist in artists:
        json_file = open('data/meta/'+artist+'.json', encoding='utf8')
        data_dict = json.load(json_file)
        img_ids = [data['contentId'] for data in data_dict]
        img_years = [data['completitionYear'] for data in data_dict]
        
        for i in range(len(img_ids)):
            img_year = str(img_years[i])
            if img_year == "None":
                img_year = "unknown-year"
            os.rename(
                "data/images/" + artist + "/" + img_year + "/" + str(img_ids[i]) + ".jpg", 
                "data/images/" + artist + "/" + str(img_ids[i]) + ".jpg"
            )

def clean_data(artists, styles):
    for artist in artists:
        json_file = open('data/meta/'+artist+'.json', encoding='utf8')
        data_dict = json.load(json_file)
        img_ids = [data['contentId'] for data in data_dict if data['style'] not in styles]
        for i in range(len(img_ids)):
            os.remove( 
                "data/images/" + artist + "/" + str(img_ids[i]) + ".jpg"
            )

def get_data_by_style(artists, styles):
    total = 0
    for artist in artists:
        json_file = open('data/meta/'+artist+'.json', encoding='utf8')
        data_dict = json.load(json_file)

        img_ids = [data['contentId'] for data in data_dict if str(data['style']) in styles]
        total += len(img_ids)
        print(len(img_ids))

        for id in img_ids:
            image = Image.open('data/images/'+artist+'/'+str(id) + ".jpg")
            width, height = image.size   # Get dimensions

            # left = (width - hp.img_size)/2
            # top = (height - hp.img_size)/2
            # right = (width + hp.img_size)/2
            # bottom = (height + hp.img_size)/2

            # # Crop the center of the image
            # image = image.crop((left, top, right, bottom))
    print(total)

def main():
    # ----------------------------------------------------------------- 5017
    # get_data_by_style(
    #     [
    #         "eugene-boudin", "james-mcneill-whistler", "vincent-van-gogh", "camille-pissarro", 
    #         "edouard-manet", "alfred-sisley", "claude-monet", "pierre-auguste-renoir", "mary-cassatt", 
    #         "giovanni-boldini", "auguste-herbin"
    #     ],
    #     ["Impressionism"])

    # ----------------------------------------------------------------- 2184
    # get_data_by_style([
    #     "paolo-uccello", "masaccio", "piero-della-francesca", "benozzo-gozzoli", "sandro-botticelli", 
    #     "filippo-lippi", "francesco-del-cossa", "andrea-mantegna", "pietro-perugino", "sebastiano-del-piombo",
    #     "leonardo-da-vinci", "michelangelo", "titian", "raphael", "giovanni-bellini", "correggio", "domenico-ghirlandaio"
    #     ], ["Early Renaissance", "High Renaissance", "Mannerism (Late Renaissance)"])

    # ----------------------------------------------------------------- 2076 
        
    # get_data_by_style([
    #     "yiannis-moralis", "andre-derain", "pablo-picasso", "paul-cezanne", "georges-braque", "juan-gris", "jean-metzinger", 
    #     "josef-capek", "kmetty-janos", "le-corbusier", "c-r-w-nevinson", "auguste-herbin", "robert-delaunay", 
    #     "kazimir-malevich", "david-kakabadze", "gosta-adrian-nilsson", "jacques-villon", "marcel-duchamp", 
    #     "raoul-dufy", "fernand-leger", "leopold-survage", "maria-blanchard",
    #     "henri-le-fauconnier", "pyotr-konchalovsky", "paul-cezanne", "amadeo-de-souza-cardoso", "bela-kadar",  "georg-pauli", "louis-marcoussis", "marevna-marie-vorobieff",
    #     "ossip-zadkine", "roger-de-la-fresnaye", "albert-gleizes", "amadeo-de-souza-cardoso", 
    #     "andr-lhote", "max-weber"], 
    #     ["Cubism", "Synthetic Cubism", "Cubism, Expressionism", "Cubism, Futurism", "Analytical Cubism", 
    #     "Art Nouveau (Modern), Cubism", "Expressionism, Cubism", "Mechanistic Cubism", "Cubism, Surrealism",
    #     "Surrealism, Cubism", "Synthetic Cubism, Cubism", "Cubism, Naïve Art (Primitivism)", "Cubism, Post-Impressionism",
    #     "Orphism", "Impressionism, Cubism", "Cubism, Tubism", "Purism"])

    # ----------------------------------------------------------------- 1971
    # get_data_by_style(["mark-tobey", "mark-rothko", "morris-louis", "robert-goodnough", 
    # "theodoros-stamos", "helen-frankenthaler", "ronnie-landfield", "brice-marden", "john-hoyland", 
    # "sam-gilliam", "jasper-johns", "walasse-ting", "cy-twombly", "joan-mitchell", "paul-jenkins", 
    # "richard-pousette-dart", "alexander-bogen", "friedel-dzubas", "philip-guston", "clyfford-still", 
    # "will-barnet", "hans-hofmann", "richard-diebenkorn", "jack-bush", "conrad-marca-relli", "adolph-gottlieb",
    # "ad-reinhardt", "barnett-newman", "beauford-delaney", "arshile-gorky"], 
    #     ["Abstract Expressionism", "Abstract Expressionism, Action painting", "Color Field Painting", "Abstract Art", 
    #     "Abstract Expressionism, Lyrical Abstraction", "Abstract Expressionism, Abstract Art", "Color Field Painting, Abstract Art",
    #     "Abstract Expressionism, Pop Art", "Abstract Expressionism, Color Field Painting", "Color Field Painting, Minimalism", 
    #     "Indian Space painting", "Expressionism, Indian Space painting", "Expressionism", "Minimalism", 
    #     "Naïve Art (Primitivism), Pop Art", "Color Field Painting, Lyrical Abstraction",
    #     ])

    # ----------------------------------------------------------------- 2260
    # get_data_by_style([
    #     "francisco-goya", "thomas-lawrence", "charles-turner", "orest-kiprensky", 
    #     "ivan-aivazovsky", "albert-bierstadt", "sir-lawrence-alma-tadema", "caspar-david-friedrich", "thomas-cole"], 
    #     ["Romanticism"])


if __name__ == '__main__':
    main()