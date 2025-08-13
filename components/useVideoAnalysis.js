// components/useVideoAnalysis.js
import { useEffect, useMemo, useRef, useState } from 'react';
import { track } from '@vercel/analytics';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || '';

export default function useVideoAnalysis() {
  const [file, setFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState('');
  const [jobId, setJobId] = useState(null);
  const [status, setStatus] = useState('idle'); // idle | ready | processing | done | failed
  const [progress, setProgress] = useState(0);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');

  const [summary, setSummary] = useState('');
  const [intuitiveScore, setIntuitiveScore] = useState(null);
  const [anomalies, setAnomalies] = useState([]);
  const [personPresent, setPersonPresent] = useState(null);
  const [thresholdScore, setThresholdScore] = useState(null);
  const [finalModelScore, setFinalModelScore] = useState(null);
  const [clips, setClips] = useState([]);
  const [frames, setFrames] = useState([]);

  const pollingRef = useRef(null);
  const isMock = !API_BASE;

  useEffect(() => () => { if (videoUrl) URL.revokeObjectURL(videoUrl); }, [videoUrl]);

  const decision = useMemo(() => {
    if (finalModelScore == null || thresholdScore == null) return null;
    return finalModelScore >= thresholdScore ? 'AI-Generated' : 'Real';
  }, [finalModelScore, thresholdScore]);

  function resetResults() {
    setJobId(null);
    setProgress(0);
    setLastUpdated(null);
    setSummary('');
    setIntuitiveScore(null);
    setAnomalies([]);
    setPersonPresent(null);
    setThresholdScore(null);
    setFinalModelScore(null);
    setClips([]);
    setFrames([]);
    setErrorMsg('');
    if (pollingRef.current) { clearInterval(pollingRef.current); pollingRef.current = null; }
  }

  function onFileChange(e) {
    const f = e.target.files?.[0];
    if (!f) return;
    setFile(f);
    setVideoUrl(URL.createObjectURL(f));
    setStatus('ready');
    resetResults();
  }

  function hardReset() {
    setFile(null);
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setVideoUrl('');
    setStatus('idle');
    resetResults();
  }

  async function handleAnalyze() {
    if (!file || status === 'processing') return;
    setErrorMsg('');
    setStatus('processing');
    setProgress(1);

    if (isMock) return mockRun();

    try {
      const fd = new FormData();
      fd.append('file', file);
      const res = await fetch(`${API_BASE}/analyze`, { method: 'POST', body: fd });
      if (!res.ok) throw new Error(`Analyze failed (${res.status})`);
      const data = await res.json();
      const jid = data.job_id;
      setJobId(jid);
      startPolling(jid);
    } catch (err) {
      setStatus('failed');
      setErrorMsg(err.message || 'Analyze request failed');
    }
  }

  function startPolling(jid) {
    if (pollingRef.current) clearInterval(pollingRef.current);
    pollingRef.current = setInterval(async () => {
      try {
        const r = await fetch(`${API_BASE}/status?job_id=${encodeURIComponent(jid)}`);
        if (!r.ok) throw new Error(`Status ${r.status}`);
        const s = await r.json();

        setProgress(Number(s?.progress ?? 0));
        setStatus(s?.status === 'done' ? 'done' : s?.status === 'failed' ? 'failed' : 'processing');
        setLastUpdated(new Date().toLocaleTimeString());

        const p = s?.partial || {};
        if (typeof p.summary === 'string') setSummary(p.summary);
        if (typeof p.intuitive_score === 'number') setIntuitiveScore(Math.round(p.intuitive_score));
        if (Array.isArray(p.anomalies)) setAnomalies(p.anomalies);
        if (typeof p.person_present === 'boolean') setPersonPresent(p.person_present);
        if (typeof p.threshold_score === 'number') setThresholdScore(p.threshold_score);
        if (typeof p.final_model_score === 'number') setFinalModelScore(p.final_model_score);

        const assets = s?.assets || {};
        if (Array.isArray(assets.clips)) setClips(assets.clips);
        if (Array.isArray(assets.frames)) setFrames(assets.frames);

        if (s?.status === 'done' || s?.status === 'failed') {
          clearInterval(pollingRef.current); pollingRef.current = null;
          if (s?.status === 'done') {
            try {
              const rr = await fetch(`${API_BASE}/result?job_id=${encodeURIComponent(jid)}`);
              if (rr.ok) {
                const fin = await rr.json();
                const fp = fin?.partial || {};
                if (typeof fp.summary === 'string') setSummary(fp.summary);
                if (typeof fp.intuitive_score === 'number') setIntuitiveScore(Math.round(fp.intuitive_score));
                if (Array.isArray(fp.anomalies)) setAnomalies(fp.anomalies);
                if (typeof fp.person_present === 'boolean') setPersonPresent(fp.person_present);
                if (typeof fp.threshold_score === 'number') setThresholdScore(fp.threshold_score);
                if (typeof fp.final_model_score === 'number') setFinalModelScore(fp.final_model_score);
                const fa = fin?.assets || {};
                if (Array.isArray(fa.clips)) setClips(fa.clips);
                if (Array.isArray(fa.frames)) setFrames(fa.frames);
                setLastUpdated(new Date().toLocaleTimeString());
              }
            } catch {}
          } else {
            setErrorMsg(s?.error || 'Processing failed');
          }
        }
      } catch (e) {
        setStatus('failed');
        setErrorMsg(e.message || 'Polling failed');
        if (pollingRef.current) { clearInterval(pollingRef.current); pollingRef.current = null; }
      }
    }, 2000);
  }

  function mockRun() {
    setProgress(8);
    setTimeout(() => {
      setSummary('A person speaks to camera indoors; occasional hand gestures; background poster visible.');
      setIntuitiveScore(67);
      setPersonPresent(true);
      setAnomalies([{ label: 'hand-morph', count: 2, timestamps: [3.2, 8.7] }]);
      setThresholdScore(0.5);
      setFinalModelScore(0.62);
      setFrames([
        { url: '/placeholder/frame1.jpg', t: 3.2, note: 'fingers look irregular' },
        { url: '/placeholder/frame2.jpg', t: 8.7, note: 'wrist contour artifact' }
      ]);
      setProgress(45);
      setLastUpdated(new Date().toLocaleTimeString());
    }, 1200);

    setTimeout(() => {
      setSummary(prev => prev + ' Lighting flicker around 12s.');
      setAnomalies(prev => [...prev, { label: 'lighting-flicker', count: 1, timestamps: [12.1] }]);
      setClips([
        { url: '/placeholder/clip1.mp4', start: 2.9, end: 3.6, caption: 'hand anomaly' },
        { url: '/placeholder/clip2.mp4', start: 11.8, end: 12.4, caption: 'lighting flicker' }
      ]);
      setProgress(78);
      setLastUpdated(new Date().toLocaleTimeString());
    }, 2600);

    setTimeout(() => {
      setFinalModelScore(0.82);
      setProgress(100);
      setStatus('done');
      setLastUpdated(new Date().toLocaleTimeString());
    }, 4200);
  }

  // --------- Vercel Analytics: Track when inference completes ---------
  useEffect(() => {
    if (status === 'done' && finalModelScore != null) {
      // Keep it anonymous: avoid filenames/PII
      track('analysis_done', {
        score: Number(finalModelScore.toFixed(3)),
        frames: Array.isArray(frames) ? frames.length : 0,
        hasClips: Array.isArray(clips) && clips.length > 0 ? 1 : 0,
        mock: isMock ? 1 : 0,
      });
    }
  }, [status, finalModelScore, frames, clips, isMock]);
  // --------------------------------------------------------------------

  return {
    file, videoUrl, status, progress, lastUpdated, isMock, errorMsg,
    summary, intuitiveScore, anomalies, personPresent, thresholdScore, finalModelScore, clips, frames,
    decision,
    onFileChange, handleAnalyze, hardReset,
  };
}
