import re
import pdftotext
import pandas as pd
from pathlib import Path

df = pd.DataFrame(columns = ['Progress Note', 'Diagnosis', 'Lab Order', 'Medication', 'Label'])
path = Path('../1/')
files = path.glob('*.pdf')
for file_ in files:
    label = file_.stem[-1:]
    with open(file_, 'rb') as f:
        pdf = pdftotext.PDF(f)
    
    text = '\n\n'.join(pdf)
    with open(f'./output/{file_.stem}.text', 'w') as f:
        f.write(text)

    lines = []
    lab = []
    progress = []
    medication = []
    diagnosis = []
    with open(f'./output/{file_.stem}.text') as f:
        for line in f.readlines():
            lines.append(str(line).strip())
    lines = [sub.replace('*** End ***', '') for sub in lines]
    lines = list(filter(lambda x: not re.match('[0-9]{2}[\-,:][0-9]{2}[\-,:][0-9]{2}', x), lines))
    for (i, line) in enumerate(lines):
        sentence = str(line).strip()
        if (sentence.startswith('UHID')):
            lines[i] = ''
    titles = ['PROGRESS NOTES Doctor','LAB ORDER Doctor','MEDICATION ORDER Doctor','DIAGNOSIS Doctor']
    for (i, line) in enumerate(lines):
        sentence = str(line).strip()
        if any(sentence.startswith(x) for x in titles):
            notes = []
            if (sentence.startswith(titles[0])):
                notes = progress
            if (sentence.startswith(titles[1])):
                notes = lab
            if (sentence.startswith(titles[2])):
                notes = medication
            if (sentence.startswith(titles[3])):
                notes = diagnosis

            valid= True
            line_index = 1
            while valid:
                try:
                    line_state = any(lines[i+line_index].startswith(a) for a in titles)
                    if line_state:
                        valid = False
                    notes.append(lines[i+ line_index])
                    line_index += 1
                except:
                    break
    lab = ''.join([' '.join(lab)])
    progress = ''.join([' '.join(progress)])
    diagnosis = ''.join([' '.join(diagnosis)])
    medication = ''.join([' '.join(medication)])
    content = {
        'Progress Note' : progress,
        'Diagnosis' : diagnosis,
        'Lab Order' : lab,
        'Medication' : medication,
        'Label' : label
    }
    df = df.append(content, ignore_index=True)
df.to_csv('steven2.csv')
print('Done')