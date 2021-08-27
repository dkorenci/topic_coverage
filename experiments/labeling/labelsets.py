'''
Central place to define sets of labels, for sample generation,
model training, mapping ...
'''
__labels_rights = [
    'civil rights movement', 'lgbt rights',  'police brutality',
    'chapel hill', 'fraternity racism', 'reproductive rights',
    'violence against women', 'death penalty', 'surveillance',
    'gun rights', 'net neutrality', 'marijuana', 'vaccination',
]

__labels_rights_final = [
    'civil rights movement', 'lgbt rights',  'police brutality',
    'chapel hill', 'reproductive rights',
    'violence against women', 'death penalty', 'surveillance',
    'gun rights', 'net neutrality', 'marijuana', 'vaccination',
]

def labels_rights():
    return list(__labels_rights)

def labels_rights_final():
    return list(__labels_rights_final)

__labels_themecov = [
    'new_themes', 'old_themes'
]

def labels_themecov():
    return list(__labels_themecov)