import genanki
import random
from cs50p_deck_full import cards

model_id = random.randrange(1 << 30, 1 << 31)

model = genanki.Model(
    model_id,
    "CS50P Deck Model",
    fields=[
        {"name": "Question"},
        {"name": "Answer"},
    ],
    templates=[
        {
            "name": "Card 1",
            "qfmt": """
<style>
pre {
  background: #1e1e1e;
  color: #d4d4d4;
  padding: 10px;
  border-radius: 6px;
  font-family: "Fira Code", monospace;
  font-size: 15px;
  white-space: pre;         /* preserves indentation and newlines */
  line-height: 1.4;
  overflow-x: auto;
}
</style>
<pre>{{Question}}</pre>
""",
            "afmt": """
{{FrontSide}}
<hr id="answer">
<pre>{{Answer}}</pre>
""",
        },
    ],
    css="""
.card {
  font-family: "Fira Code", monospace;
  font-size: 15px;
  text-align: left;
  color: #222;
  background-color: #fdfdfd;
  white-space: pre-wrap;
}
pre {
  background: #1e1e1e;
  color: #d4d4d4;
  padding: 8px;
  border-radius: 6px;
  overflow-x: auto;
}
""",
)

deck_id = random.randrange(1 << 30, 1 << 31)
deck = genanki.Deck(deck_id, "CS50P Python Deck (Tagged)")

for q, a, tag in cards:
    note = genanki.Note(
        model=model,
        fields=[q, a],
        tags=[tag]
    )
    deck.add_note(note)

package = genanki.Package(deck)
package.output_file = "cs50p_deck_tagged.apkg"
package.write_to_file("cs50p_deck_tagged.apkg")

print("✅ Tagged deck built -> cs50p_deck_tagged.apkg")
