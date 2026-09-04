import React, { useState, useRef } from 'react';
import { pipeline } from '@huggingface/transformers';

const DEFAULT_ARABIC = `في السنوات الأخيرة، أصبح الذكاء الاصطناعي من أكثر التقنيات تأثيرًا في العالم.
تستخدم الشركات هذه التكنولوجيا لتحليل البيانات وتحسين الخدمات وتوفير الوقت والمال.
كما أصبحت تطبيقات الذكاء الاصطناعي موجودة في التعليم والطب والترجمة وحتى في حياتنا اليومية.
ومع ذلك، فإن استخدام هذه التقنيات يحتاج إلى مسؤولية كبيرة، لأن القرارات التي تتخذها الأنظمة الذكية قد تؤثر في حياة الناس.
لذلك من المهم تطوير هذه الأنظمة بطريقة آمنة وشفافة تحافظ على خصوصية المستخدمين.
في المستقبل، من المتوقع أن تصبح أدوات الذكاء الاصطناعي أكثر قدرة على فهم اللغة والصور والصوت.
وقد يساعد ذلك الناس على التواصل مع بعضهم البعض بسهولة، حتى عندما يتحدثون لغات مختلفة.
لكن التكنولوجيا وحدها لا تكفي، فالإنسان سيظل مسؤولًا عن كيفية استخدامها والأهداف التي يريد تحقيقها.`;

let translatorPromise = null;

function getTranslator(onProgress) {
  if (!translatorPromise) {
    translatorPromise = pipeline('translation', 'Xenova/opus-mt-ar-en', {
      dtype: {
        encoder_model: 'fp32',
        decoder_model_merged: 'fp32',
      },
      progress_callback: onProgress,
    });
  }
  return translatorPromise;
}

function splitIntoSentences(text) {
  return text
    .split(/(?<=[.!?؟\n])\s*/)
    .map((s) => s.trim())
    .filter(Boolean);
}

// Cleans punctuation token artifacts (e.g. "people ' s" -> "people's")
function cleanPunctuation(text) {
  return text
    .replace(/\s+([',.?!:;])/g, '$1')
    .replace(/([(\[{])\s+/g, '$1')
    .replace(/\s+/g, ' ')
    .trim();
}

export default function App() {
  const [input, setInput] = useState(DEFAULT_ARABIC);
  const [translation, setTranslation] = useState('');
  const [status, setStatus] = useState('');
  const [loading, setLoading] = useState(false);
  const modelReady = useRef(false);

  const handleTranslate = async () => {
    if (!input.trim()) return;
    setLoading(true);
    setTranslation('');

    try {
      setStatus(
        modelReady.current
          ? 'Translating...'
          : 'Loading model (first run only, then cached)...'
      );

      const translator = await getTranslator((progress) => {
        if (progress.status === 'progress' && !modelReady.current) {
          setStatus(`Downloading model: ${Math.round(progress.progress || 0)}%`);
        }
      });

      modelReady.current = true;
      const sentences = splitIntoSentences(input);
      const results = [];

      for (let i = 0; i < sentences.length; i++) {
        setStatus(`Translating sentence ${i + 1} of ${sentences.length}...`);

        // Beam search prevents literal collapse like "Artificial IQ"
        const output = await translator(sentences[i], {
          num_beams: 4,
          max_new_tokens: 256,
        });

        const cleaned = cleanPunctuation(output[0].translation_text);
        results.push(cleaned);
        setTranslation(results.join(' '));
      }

      setStatus('Done!');
    } catch (err) {
      console.error(err);
      setStatus('Translation failed. Check console for details.');
    } finally {
      setLoading(false);
    }
  };

  const handleReset = () => {
    setInput(DEFAULT_ARABIC);
    setTranslation('');
    setStatus('');
  };

  return (
    <div style={{ maxWidth: '680px', margin: '2rem auto', padding: '0 1rem', fontFamily: 'system-ui, sans-serif' }}>
      <h2>In-Browser Arabic &rarr; English Translation</h2>

      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '6px' }}>
        <label htmlFor="arabic-input" style={{ fontWeight: '600', fontSize: '0.95rem' }}>
          Arabic Text:
        </label>
        <button
          type="button"
          onClick={handleReset}
          disabled={loading}
          style={{ background: 'none', border: 'none', color: '#0066cc', cursor: 'pointer', fontSize: '0.85rem' }}
        >
          Reset to default
        </button>
      </div>

      <textarea
        id="arabic-input"
        dir="rtl"
        value={input}
        onChange={(e) => setInput(e.target.value)}
        rows={8}
        placeholder="اكتب النص العربي هنا..."
        style={{
          width: '100%',
          fontSize: '1.05rem',
          lineHeight: '1.6',
          padding: '10px',
          borderRadius: '6px',
          border: '1px solid #ccc',
          boxSizing: 'border-box',
          resize: 'vertical',
        }}
      />

      <div style={{ marginTop: '12px', display: 'flex', gap: '10px' }}>
        <button
          onClick={handleTranslate}
          disabled={loading || !input.trim()}
          style={{
            padding: '10px 20px',
            backgroundColor: loading ? '#9e9e9e' : '#0066cc',
            color: '#fff',
            border: 'none',
            borderRadius: '4px',
            cursor: loading ? 'not-allowed' : 'pointer',
            fontWeight: '600',
          }}
        >
          {loading ? 'Translating...' : 'Translate to English'}
        </button>
      </div>

      {status && (
        <p style={{ color: '#555', fontSize: '0.9rem', marginTop: '12px' }}>
          <strong>Status:</strong> {status}
        </p>
      )}

      {translation && (
        <div
          style={{
            marginTop: '1.5rem',
            padding: '1rem',
            backgroundColor: '#f0f7ff',
            border: '1px solid #cce3f5',
            borderRadius: '6px',
          }}
        >
          <strong style={{ display: 'block', marginBottom: '6px', color: '#004085' }}>
            English Translation:
          </strong>
          <p style={{ margin: 0, fontSize: '1rem', lineHeight: '1.6', color: '#222' }}>
            {translation}
          </p>
        </div>
      )}
    </div>
  );
}