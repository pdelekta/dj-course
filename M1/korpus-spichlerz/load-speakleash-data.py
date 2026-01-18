import os
from speakleash import Speakleash


def get_data(txt_folder_name):
    """
    This method creates the necessary directories if they don't already exist.

    :param txt_folder_name: The name of the folder where the high-quality documents will be saved.
    :return: None
    """
    base_dir = os.path.join(".")
    speakleash_dir = os.path.join(base_dir, "datasets")
    replicate_to_txt = os.path.join(base_dir, txt_folder_name)

    if not os.path.exists(speakleash_dir):
        os.makedirs(speakleash_dir)
    if not os.path.exists(replicate_to_txt):
        os.makedirs(replicate_to_txt)

def save_quality_docs(txt_folder_name, quality='HIGH'):
    """
    This method saves documents of the specified quality from the Speakleash dataset to a specified folder.

    :param txt_folder_name: The name of the folder where the documents will be saved.
    :param quality: The judge quality of the documents.
    :return: None
    """
    base_dir = os.path.join(".")
    speakleash_dir = os.path.join(base_dir, "datasets")
    replicate_to_txt = os.path.join(base_dir, txt_folder_name)

    sl = Speakleash(speakleash_dir)
    name = "plwiki"
    limit = 80
    counter = 0
    ds = sl.get(name).ext_data

    for doc in ds:
        txt, meta = doc
        if meta.get("quality", "") == quality:
            print(f"{quality}-quality document")
            with open(os.path.join(replicate_to_txt, f'{quality}_quality_doc_{counter}.txt'), 'w', encoding='utf-8') as out_file:
                out_file.write(txt)
            counter += 1
            if counter > limit:
                break


if __name__ == "__main__":
    txt_folder_name = "output"
    get_data(txt_folder_name)
    save_quality_docs(txt_folder_name, "HIGH")