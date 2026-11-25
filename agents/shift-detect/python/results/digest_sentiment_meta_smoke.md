# Digest pipeline — `sentiment_meta`

ρ_PO = 0.256
Top block: `spacy_style`

Driving subset: ['vader_compound', 'vader_neg', 'vader_polar_gap', 'vader_neu']

Sentiment-change drivers (vader_sent subset) = ['vader_compound', 'vader_neg', 'vader_polar_gap', 'vader_neu']; global carriers = ['vader_compound', 'vader_neg', 'vader_polar_gap', 'vader_neu', 'n_tokens'].

## Blocks

- spacy_style: imp=0, p=1.000
- spacy_syntax: imp=0, p=1.000
- spacy_ner: imp=0, p=1.000
- vader_sent: imp=0, p=1.000
- discourse: imp=0, p=1.000
- structure_cues: imp=0, p=1.000
