def month_clean(row):
    if isinstance(row, str):
        row = row.lower()
        for s in Seperator:
            if s in row:
                return row.split(s)[-1].strip().lower()
        
        for m in Month:
            if m[:3] in row:
                return m
        
        for s in Season:
            if s in row:
                return Month[Season.index(s) * 3 + 2]
        
        for q in Quarter:
            if q in row:
                return Month[Quarter.index(q) * 3]
        
        return row.lower()
    else:
        return random.choice(Month)

def split_author(row):
    try:
        return row.split("|")
    except:
        return row

def author_permutations(row):
    if isinstance(row, list):
        return list(permutations(row, 2))
    else:
        return row