"""Static semantics and HTTP asset contracts; not a visual browser test."""
from html.parser import HTMLParser
from pathlib import Path
import unittest
from test_intent_flow import request

class Document(HTMLParser):
    def __init__(self,text):
        super().__init__();self.nodes=[];self.feed(text)
    def handle_starttag(self,tag,attrs):self.nodes.append((tag,dict(attrs)))

class AccessibleUI(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root=Path(__file__).resolve().parent
        cls.doc=Document((cls.root/'static/index.html').read_text(encoding='utf-8'))
    def test_form_controls_have_associated_labels(self):
        labels={a.get('for') for tag,a in self.doc.nodes if tag=='label'}
        for tag,a in self.doc.nodes:
            if tag in ('select','textarea'):
                self.assertIn(a.get('id'),labels,a)
    def test_ids_unique_and_accessibility_references_resolve(self):
        ids=[a['id'] for _,a in self.doc.nodes if 'id' in a]
        self.assertEqual(len(ids),len(set(ids)))
        for _,a in self.doc.nodes:
            for key in ('aria-labelledby','aria-describedby','aria-controls'):
                for value in a.get(key,'').split():self.assertIn(value,ids)
    def test_local_dependencies_and_native_disclosures(self):
        scripts=[a.get('src') for tag,a in self.doc.nodes if tag=='script']
        self.assertEqual(scripts,['/presentation.js','/recorder.js','/app.js'])
        by_id={a['id']:(tag,a) for tag,a in self.doc.nodes if 'id' in a}
        for key in ('inputDetails','modelDetails','reasoningDetails','conversationDetails','connectionDetails'):
            self.assertEqual(by_id[key][0],'details')
        self.assertIn('open',by_id['inputDetails'][1])
        self.assertNotIn('open',by_id['modelDetails'][1])
    def test_runtime_assets_served_without_new_network_origins(self):
        for path in ('/presentation.js','/icons.svg'):
            code,body=request(path,method='GET')
            self.assertEqual(code,200);self.assertTrue(body)
        code,_=request('/static/vendor/not-an-endpoint',method='GET');self.assertEqual(code,404)
    def test_browser_version_matches_backend(self):
        from app_version import APP_VERSION
        body=next(a for tag,a in self.doc.nodes if tag=='body')
        self.assertEqual(body['data-app-version'],APP_VERSION)
if __name__=='__main__':unittest.main()
