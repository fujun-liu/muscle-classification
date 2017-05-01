'''
    find whole slide image labels
'''
import pandas as pd
import numpy as np

def normalize_name(name):
    patterns = {' ':'', '-':'', '_':''}
    for k,v in patterns.items():
        name = name.replace(k, v)
    return name.lower()

class LabelParser():
    def __init__(self, label_excel_path='slide_labels.xlsx'):
        df = pd.read_excel(label_excel_path)
        num_classes = 0
        slide_label_dict = dict()
        for index, row in df.iterrows():
            name = str(row['Slide Name'])
            label = row['Diagnosis']
            num_classes = max(num_classes, label)
            slide_label_dict[normalize_name(name)] = label-1
        self.slide_label_dict = slide_label_dict
        self.num_classes = num_classes
        print len(self.slide_label_dict)
        cnt = np.zeros(num_classes)
        for k, v in slide_label_dict.items():
            cnt[v] += 1
        for i in range(num_classes):
            print 'class {}: {}'.format(i, cnt[i])

    def __call__(self, slide_name):
        # preprocess image name
        slide_name = normalize_name(slide_name)
        for k, v in self.slide_label_dict.items():
            if k in slide_name:
                return v, k
        return None, None

def test():
    label_parser = LabelParser()

if __name__ == "__main__":
    test()
