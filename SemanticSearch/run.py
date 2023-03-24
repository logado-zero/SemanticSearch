import pandas as pd
from module.semantic_search import Semantic_Search

if __name__=="__main__":
    #Load dataset
    file = 'dataset/ATTRIBUTE_DIC_BluePrint.xls'

    data = pd.read_excel(file)

    df = data[['ATTRIBUTE','DESCRIPTION']]
    del data
    # Prepare for dataset
    df['ATTRI_DES'] = "Attribute is "+ df['ATTRIBUTE'] + " and description is " + df['DESCRIPTION']

    # Load model
    model = Semantic_Search(model_name = "model/all-mpnet-base-v2", available_dataset="dataset/dataset.index")

    # Process dataset
    model.process_dataset(df.ATTRI_DES.tolist())

    while True:
        attri = input("Attribute Search: ")
        des = input("Description Search: ")
        if attri == "out" or des == "out":
            print("End Search")
            break

        sen_search = "Attribute is " + attri + " and description is " + des

        results = model.search(query = sen_search, dataset= df, name_content= ["ATTRIBUTE","DESCRIPTION"])

        print("\n")
        print("#######      Result Search       ########")
        for res in results:
            res_attr, res_des, res_dis = res["ATTRIBUTE"], res["DESCRIPTION"], res["DISTANCE"]
            print("_"*20)
            print(f"\nAttribute: {res_attr}")
            print(f"Description:\n \t {res_des}")
            print(f"Score: {res_dis}")
    