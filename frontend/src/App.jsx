import { useEffect, useState } from 'react';
import { ArrowUpRight, BrainCircuit, ChevronDown, FileText, Gauge, Layers3, MessageSquare, Search, Send, Trash2, Upload, Zap } from 'lucide-react';

const API = `${import.meta.env.VITE_API_URL || 'http://localhost:8000'}/api`;

const PIPELINE_STAGES = [
  { title: '1. Extract', detail: 'pypdf · per page' },
  { title: '2. Normalise', detail: 'repair the text' },
  { title: '3. Chunk', detail: null },
  { title: '4. Embed', detail: 'MiniLM · 384d · local' },
  { title: '5. Store', detail: 'qdrant · cosine' },
];

const SUGGESTED_QUESTIONS = [
  'What are the security requirements?',
  'How does single sign-on work?',
  'What are the performance targets?',
  'What is out of scope for this release?',
];

function scoreLabel(score) {
  return `${Math.round(Math.max(0, score) * 100)}% match`;
}

export default function App() {
  const [tab, setTab] = useState('ingest');
  const [stats, setStats] = useState(null);
  const [chunks, setChunks] = useState([]);
  const [notice, setNotice] = useState('');
  const [busy, setBusy] = useState(false);

  const [chunkSize, setChunkSize] = useState(180);
  const [chunkOverlap, setChunkOverlap] = useState(40);

  const [query, setQuery] = useState('What is the purpose of the VWO Login Dashboard?');
  const [results, setResults] = useState([]);
  const [answer, setAnswer] = useState('');
  const [searchMetrics, setSearchMetrics] = useState(null);

  const [chatQuestion, setChatQuestion] = useState('What are the security requirements?');
  const [chatAnswer, setChatAnswer] = useState('');
  const [chatSources, setChatSources] = useState([]);
  const [chatMetrics, setChatMetrics] = useState(null);
  const [chatAskedAt, setChatAskedAt] = useState(null);

  async function loadData() {
    const [statsResponse, chunksResponse] = await Promise.all([fetch(`${API}/stats`), fetch(`${API}/chunks`)]);
    const statsData = await statsResponse.json();
    setStats(statsData);
    setChunks((await chunksResponse.json()).chunks);
    if (statsData.chunk_size_words) setChunkSize(statsData.chunk_size_words);
    if (statsData.chunk_overlap_words) setChunkOverlap(statsData.chunk_overlap_words);
  }

  useEffect(() => { loadData().catch(() => setNotice('Start the API with uvicorn to connect the explorer.')); }, []);

  async function runQuery(question) {
    const response = await fetch(`${API}/search`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ query: question, top_k: 3 }) });
    const data = await response.json();
    if (!response.ok) throw new Error(data.detail || 'Search failed');
    return data;
  }

  async function search() {
    if (!query.trim()) return;
    setBusy(true);
    setNotice('');
    try {
      const data = await runQuery(query);
      setResults(data.results);
      setAnswer(data.answer || 'No Groq answer returned. The retrieved evidence is ready below.');
      setSearchMetrics(data.metrics || null);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function askChat(question) {
    const trimmed = (question ?? chatQuestion).trim();
    if (!trimmed) return;
    setChatQuestion(trimmed);
    setBusy(true);
    setNotice('');
    try {
      const data = await runQuery(trimmed);
      setChatAnswer(data.answer || 'No Groq answer returned. See the retrieved chunks below.');
      setChatSources(data.results || []);
      setChatMetrics(data.metrics || null);
      setChatAskedAt(new Date());
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function ingest(event) {
    const files = Array.from(event.target.files || []);
    if (files.length === 0) return;
    event.target.value = '';
    setNotice(files.length > 1 ? `Reading ${files.length} PDFs, splitting chunks, and building the index...` : 'Reading PDF, splitting chunks, and building the index...');
    setBusy(true);
    const body = new FormData();
    files.forEach((file) => body.append('files', file));
    try {
      const response = await fetch(`${API}/ingest?chunk_size=${chunkSize}&chunk_overlap=${chunkOverlap}`, { method: 'POST', body });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Ingestion failed');
      await loadData();
      setNotice(`${data.document_count} document(s) indexed, ${data.chunks} chunks total.`);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function removeDocument(name) {
    setBusy(true);
    setNotice(`Removing ${name}...`);
    try {
      const response = await fetch(`${API}/documents/${encodeURIComponent(name)}`, { method: 'DELETE' });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Could not remove document');
      await loadData();
      setNotice(`${name} removed. ${data.document_count} document(s) remaining.`);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  async function reindex() {
    setNotice('Re-chunking and re-embedding with the new settings...');
    setBusy(true);
    try {
      const response = await fetch(`${API}/reindex`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ chunk_size: chunkSize, chunk_overlap: chunkOverlap }) });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Reindex failed');
      await loadData();
      setNotice(`${data.document} re-indexed with ${data.chunks} chunks.`);
    } catch (error) {
      setNotice(error.message);
    } finally {
      setBusy(false);
    }
  }

  return (
    <main className="shell">
      <header className="topbar">
        <div className="brand"><span className="brand-mark"><BrainCircuit size={19} /></span><span>RAG / explorer</span></div>
        <div className="topbar-meta"><span className="live-dot" /> local pipeline <span className="slash">/</span> groq powered</div>
      </header>

      <section className="intro">
        <div className="eyebrow"><Zap size={14} /> document intelligence lab</div>
        <h1>See how a question<br /><em>travels through</em> a PDF.</h1>
        <p className="lede">RAG Explorer &middot; PDF &rarr; chunks &rarr; vectors &rarr; answers</p>
        <div className="model-badges">
          <span className="badge">{stats?.embedding_model?.split('/').pop() || 'all-MiniLM-L6-v2'}</span>
          <span className="badge">{stats?.generation_model || 'openai/gpt-oss-120b'}</span>
          <span className="badge">qdrant</span>
        </div>
        <label className="upload-button"><Upload size={16} /> add PDF(s)<input type="file" accept="application/pdf" multiple onChange={ingest} /></label>
      </section>

      <section className="stats-grid">
        <Metric icon={<FileText />} label="source documents" value={stats?.document || 'waiting...'} detail={stats ? `${stats.pages} pages indexed` : 'connect API'} />
        <Metric icon={<Layers3 />} label="knowledge chunks" value={stats?.chunks ?? '--'} detail={stats ? `${stats.chunk_size_words} words / ${stats.chunk_overlap_words} overlap` : 'not indexed'} />
        <Metric icon={<Gauge />} label="vector space" value={stats?.embedding_dimensions ? `${stats.embedding_dimensions}D` : '--'} detail={stats?.embedding_model?.split('/').pop() || 'embedding model'} />
        <Metric icon={<BrainCircuit />} label="generation" value="GPT-120B" detail={stats?.generation_model || 'Groq'} />
      </section>

      <nav className="tabs">
        <button className={tab === 'ingest' ? 'tab active' : 'tab'} onClick={() => setTab('ingest')}>1 &middot; Ingest &amp; Chunks</button>
        <button className={tab === 'search' ? 'tab active' : 'tab'} onClick={() => setTab('search')}>2 &middot; Search</button>
        <button className={tab === 'chat' ? 'tab active' : 'tab'} onClick={() => setTab('chat')}>3 &middot; Chat</button>
      </nav>

      {notice && <div className="notice">{notice}</div>}

      {tab === 'ingest' && (
        <section className="ingest-section">
          <div className="panel-heading"><div><span className="section-kicker">ingest</span><h2>Ingest the PDF</h2></div></div>
          <p className="section-desc">Five stages turn a PDF into something searchable. Upload one or more PDFs — they're merged into a single searchable index. Change the chunk settings and re-run to watch the trade-off move.</p>

          {stats?.documents?.length > 0 && (
            <div className="documents-list">
              {stats.documents.map((doc) => (
                <div className="document-chip" key={doc.name}>
                  <FileText size={13} />
                  <span>{doc.name}</span>
                  <small>{doc.pages}p</small>
                  <button onClick={() => removeDocument(doc.name)} disabled={busy} title={`Remove ${doc.name}`}><Trash2 size={13} /></button>
                </div>
              ))}
            </div>
          )}

          <div className="pipeline-row">
            {PIPELINE_STAGES.map((stage) => (
              <div className="pipeline-card" key={stage.title}>
                <strong>{stage.title}</strong>
                <span>{stage.title === '3. Chunk' ? `${chunkSize}w / ${chunkOverlap} overlap` : stage.detail}</span>
              </div>
            ))}
          </div>

          <div className="sliders-row">
            <div className="slider-block">
              <label>Chunk size: {chunkSize} words</label>
              <input type="range" min="60" max="400" step="10" value={chunkSize} onChange={(event) => setChunkSize(Number(event.target.value))} />
            </div>
            <div className="slider-block">
              <label>Overlap: {chunkOverlap} words ({Math.round((chunkOverlap / chunkSize) * 100)}%)</label>
              <input type="range" min="0" max={Math.max(0, chunkSize - 10)} step="5" value={chunkOverlap} onChange={(event) => setChunkOverlap(Number(event.target.value))} />
            </div>
          </div>

          <button className="primary-button" onClick={reindex} disabled={busy}>{busy ? 'ingesting...' : 'Ingest PDF'}</button>

          <div className="chunks-table">
            <div className="table-head"><span>chunk</span><span>document</span><span>page</span><span>size</span><span>preview</span><span /></div>
            {chunks.slice(0, 12).map((chunk) => (
              <details key={chunk.id} className="chunk-row">
                <summary><span>#{String(chunk.id).padStart(2, '0')}</span><span className="chunk-doc-name" title={chunk.document}>{chunk.document}</span><span>{chunk.page}</span><span>{chunk.word_count} words</span><span>{chunk.text.slice(0, 105)}...</span><ChevronDown size={16} /></summary>
                <div className="chunk-detail">
                  <ChunkText chunk={chunk} />
                  <div className="chunk-overlap-legend">
                    {chunk.overlap_prev_words > 0 && <span className="overlap-chip prev"><span className="swatch" /> {chunk.overlap_prev_words} words shared with chunk #{String(chunk.id - 1).padStart(2, '0')}</span>}
                    {chunk.overlap_next_words > 0 && <span className="overlap-chip next"><span className="swatch" /> {chunk.overlap_next_words} words shared with chunk #{String(chunk.id + 1).padStart(2, '0')}</span>}
                    {!chunk.overlap_prev_words && !chunk.overlap_next_words && <span className="overlap-chip none">no overlap on this chunk</span>}
                  </div>
                  <div className="chunk-vector-info">
                    <span className="section-kicker">vector store</span>
                    <div className="vector-grid">
                      <span>collection <strong>{chunk.collection}</strong></span>
                      <span>point id <strong>#{chunk.vector_id}</strong></span>
                      <span>dimensions <strong>{chunk.vector_dims}D</strong></span>
                    </div>
                    <code className="vector-preview">[{chunk.vector_preview?.join(', ')}, ...]</code>
                  </div>
                </div>
              </details>
            ))}
          </div>
        </section>
      )}

      {tab === 'search' && (
        <section className="workspace">
          <div className="search-panel panel">
            <div className="panel-heading"><div><span className="section-kicker">01 / retrieve</span><h2>Ask the document</h2></div><span className="step-number">01</span></div>
            <div className="search-box"><Search size={18} /><input value={query} onChange={(event) => setQuery(event.target.value)} onKeyDown={(event) => event.key === 'Enter' && search()} placeholder="Ask anything about the PDF..." /><button onClick={search} disabled={busy}>{busy ? 'searching' : 'run search'} <ArrowUpRight size={16} /></button></div>
            {searchMetrics && <MetricsRow metrics={searchMetrics} />}
            <div className="answer"><div className="answer-label"><span className="answer-dot" /> grounded answer</div>{answer ? <MarkdownAnswer text={answer} sources={results} /> : <p>Ask a question to see the answer and the exact chunks that support it.</p>}</div>
          </div>

          <div className="results-panel panel">
            <div className="panel-heading"><div><span className="section-kicker">02 / rank</span><h2>Top 3 retrieved chunks</h2></div><span className="retrieval-note">cosine similarity</span></div>
            {results.length === 0 ? <EmptyState /> : (
              <div className="results-list">
                {results.map((result, index) => (
                  <article className="result" key={result.id}>
                    <div className="result-top"><span className="rank">0{index + 1}</span><span className="chunk-id">chunk_{String(result.id).padStart(2, '0')} / {result.document} / p.{result.page}</span><span className="score">{scoreLabel(result.score)}</span></div>
                    <p>{result.text}</p>
                    <div className="result-footer"><span>{result.word_count} words</span><span>score {result.score.toFixed(4)}</span></div>
                  </article>
                ))}
              </div>
            )}
          </div>
        </section>
      )}

      {tab === 'chat' && (
        <section className="chat-section">
          <div className="panel-heading"><div><span className="section-kicker">03 / converse</span><h2>Retrieval + generation</h2></div></div>
          <p className="section-desc">Same retrieval as tab 2, then the top chunks are pasted into a prompt and sent to {stats?.generation_model || 'gpt-oss-120b'} on Groq. The model is told to answer only from those chunks.</p>

          <div className="search-box chat-input-box">
            <MessageSquare size={18} />
            <input value={chatQuestion} onChange={(event) => setChatQuestion(event.target.value)} onKeyDown={(event) => event.key === 'Enter' && askChat()} placeholder="Ask a question about the PDF..." />
            <button onClick={() => askChat()} disabled={busy}>{busy ? 'asking...' : 'ask'} <Send size={16} /></button>
          </div>

          <div className="suggestion-row">
            {SUGGESTED_QUESTIONS.map((question) => (
              <button key={question} className="suggestion-chip" onClick={() => askChat(question)} disabled={busy}>{question}</button>
            ))}
          </div>

          {chatMetrics && <MetricsRow metrics={chatMetrics} />}

          <div className="panel-heading answer-heading"><h3>Answer</h3>{chatAskedAt && <span className="retrieval-note">{chatAskedAt.toLocaleTimeString()}</span>}</div>
          <div className="answer chat-answer">
            {chatAnswer ? <MarkdownAnswer text={chatAnswer} sources={chatSources} /> : (
              <div className="empty"><div className="empty-icon"><MessageSquare size={20} /></div><p>Ask a question to see a grounded answer.</p><span>Each bullet is cited back to the chunk it came from.</span></div>
            )}
          </div>
        </section>
      )}

      <footer><span>RAG explorer / local demo</span><span>embeddings &rarr; vector index &rarr; top-k &rarr; LLM</span></footer>
    </main>
  );
}

function Metric({ icon, label, value, detail }) { return <div className="metric"><div className="metric-icon">{icon}</div><div><span>{label}</span><strong>{value}</strong><small>{detail}</small></div></div>; }

function MetricsRow({ metrics }) {
  const tiles = [
    { label: 'prompt tokens', value: metrics.prompt_tokens ?? '--', unit: '' },
    { label: 'output tokens', value: metrics.output_tokens ?? '--', unit: '' },
    { label: 'llm', value: metrics.llm_ms ?? '--', unit: 'ms' },
    { label: 'embed', value: metrics.embed_ms ?? '--', unit: 'ms' },
    { label: 'search', value: metrics.search_ms ?? '--', unit: 'ms' },
  ];
  return (
    <div className="metrics-row">
      {tiles.map((tile) => (
        <div className="metric-tile" key={tile.label}>
          <strong>{tile.value}{tile.unit && <small className="metric-unit">{tile.unit}</small>}</strong>
          <span>{tile.label}</span>
        </div>
      ))}
    </div>
  );
}

function renderInline(text, keyPrefix) {
  const parts = text.split(/(\*\*.+?\*\*)/g).filter(Boolean);
  return parts.map((part, index) => {
    if (part.startsWith('**') && part.endsWith('**')) {
      return <strong key={`${keyPrefix}-${index}`}>{part.slice(2, -2)}</strong>;
    }
    return <span key={`${keyPrefix}-${index}`}>{part}</span>;
  });
}

function CitationChips({ ids, sources }) {
  if (ids.length === 0) return null;
  const sourceLookup = new Map((sources || []).map((result) => [String(result.id), result]));
  return (
    <span className="citation-chips">
      {ids.map((id, index) => {
        const source = sourceLookup.get(String(id));
        return <span key={`${id}-${index}`} className="citation-chip">chunk {id}{source ? ` · ${source.document} · p.${source.page}` : ''}</span>;
      })}
    </span>
  );
}

function parseAnswerLine(rawLine) {
  const citationBrackets = [...rawLine.matchAll(/\[chunk[^\]]*\]/gi)];
  const citationIds = [...new Set(citationBrackets.flatMap((bracket) => [...bracket[0].matchAll(/\d+/g)].map((match) => match[0])))];
  const cleanLine = rawLine.replace(/\[chunk[^\]]*\]/gi, '').replace(/\s+/g, ' ').trim();
  const bulletMatch = cleanLine.match(/^[-*•]\s+(.*)/);
  return {
    isBullet: Boolean(bulletMatch),
    content: bulletMatch ? bulletMatch[1] : cleanLine,
    citationIds,
  };
}

function MarkdownAnswer({ text, sources }) {
  const lines = text.split('\n').map((line) => line.trim()).filter(Boolean);
  const blocks = [];
  let currentList = null;

  for (const rawLine of lines) {
    const parsed = parseAnswerLine(rawLine);
    if (!parsed.content) continue;
    if (parsed.isBullet) {
      if (!currentList) {
        currentList = { type: 'list', items: [] };
        blocks.push(currentList);
      }
      currentList.items.push(parsed);
    } else {
      currentList = null;
      blocks.push({ type: 'paragraph', ...parsed });
    }
  }

  return (
    <div className="markdown-answer">
      {blocks.map((block, index) => {
        if (block.type === 'list') {
          return (
            <ul key={index}>
              {block.items.map((item, itemIndex) => (
                <li key={itemIndex}>
                  {renderInline(item.content, `li-${index}-${itemIndex}`)}
                  <CitationChips ids={item.citationIds} sources={sources} />
                </li>
              ))}
            </ul>
          );
        }
        return (
          <p key={index}>
            {renderInline(block.content, `p-${index}`)}
            <CitationChips ids={block.citationIds} sources={sources} />
          </p>
        );
      })}
    </div>
  );
}

function ChunkText({ chunk }) {
  const words = chunk.text.split(' ');
  const prevCount = chunk.overlap_prev_words || 0;
  const nextCount = chunk.overlap_next_words || 0;
  const prefix = words.slice(0, prevCount);
  const suffix = nextCount > 0 ? words.slice(words.length - nextCount) : [];
  const middleEnd = nextCount > 0 ? words.length - nextCount : words.length;
  const middle = words.slice(prevCount, middleEnd);
  return (
    <p className="chunk-text">
      {prefix.length > 0 && <span className="overlap-span prev">{prefix.join(' ')} </span>}
      {middle.join(' ')}
      {suffix.length > 0 && <span className="overlap-span next"> {suffix.join(' ')}</span>}
    </p>
  );
}
function EmptyState() { return <div className="empty"><div className="empty-icon"><Search size={20} /></div><p>Your top three matches will appear here.</p><span>Each result shows its chunk id, page, similarity score, and source text.</span></div>; }
