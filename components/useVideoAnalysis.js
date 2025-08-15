// components/useVideoAnalysis.js
import { useCallback, useEffect, useRef, useState } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || '';
const POLL_INTERVAL_MS = 900;

export default function useVideoAnalysis() {
  const [file, setFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState('');
  const [status, setStatus] = useState('idle'); // idle | ready | processing | done | failed
  const [progress, setProgress] = useState(0);
  const [lastUpdated, setLastUpdated] = useState(null); // <-- number | null
  const [errorMsg, setErrorMsg] = useState('');

  const [summaryTimeline, setSummaryTimeline] = useState([]);
  const [thresholdScore, setThresholdScore] = useState(0.5);
  const [finalModelScore, setFinalModelScore] = useState(null);

  const [intuitionScore, setIntuitionScore] = useState(null);        // 0..100
  const [intuitionSegments, setIntuitionSegments] = useState([]);    // list of dicts

  const jobIdRef = useRef(null);
  const pollAbortRef = useRef(null);
  const isMock = !API_BASE;

  const getJSON = useCallback(async (url) => {
    const res = await fetch(url, {
      method: 'GET',
      cache: 'no-store',
      headers: { 'Cache-Control': 'no-cache', Pragma: 'no-cache' },
    });
    if (!res.ok) {
      const txt = await res.text().catch(() => '');
      throw new Error(`GET ${url} → ${res.status}: ${txt}`);
    }
    return res.json();
  }, []);

  const fetchStatus = useCallback(async (jid) => {
    const url = new URL(`${API_BASE}/status`);
    url.searchParams.set('job_id', jid);
    url.searchParams.set('_', Date.now().toString());
    return getJSON(url.toString());
  }, [getJSON]);

  const fetchResult = useCallback(async (jid) => {
    const url = new URL(`${API_BASE}/result`);
    url.searchParams.set('job_id', jid);
    url.searchParams.set('_', Date.now().toString());
    return getJSON(url.toString());
  }, [getJSON]);

  useEffect(() => {
    return () => {
      if (videoUrl) URL.revokeObjectURL(videoUrl);
      if (pollAbortRef.current) pollAbortRef.current.abort();
    };
  }, [videoUrl]);

  const onFileChange = useCallback((e) => {
    const f = e?.target?.files?.[0];
    if (!f) return;
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setFile(f);
    setVideoUrl(URL.createObjectURL(f));
    setStatus('ready');
    setProgress(0);
    setErrorMsg('');
    setSummaryTimeline([]);
    setFinalModelScore(null);
    setThresholdScore(0.5);
    setLastUpdated(null);
    jobIdRef.current = null;
  }, [videoUrl]);

  const hardReset = useCallback(() => {
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setFile(null);
    setVideoUrl('');
    setStatus('idle');
    setProgress(0);
    setLastUpdated(null);
    setErrorMsg('');
    setSummaryTimeline([]);
    setFinalModelScore(null);
    setThresholdScore(0.5);
    if (pollAbortRef.current) pollAbortRef.current.abort();
    jobIdRef.current = null;
  }, [videoUrl]);

  const handleAnalyze = useCallback(async () => {
    if (isMock) {
      setErrorMsg('Backend not configured. Set NEXT_PUBLIC_API_BASE_URL to your API.');
      return;
    }
    if (!file) return;

    try {
      if (pollAbortRef.current) pollAbortRef.current.abort();

      setStatus('processing');
      setProgress(5);
      setErrorMsg('');
      setLastUpdated(Date.now());     // <-- numeric

      const form = new FormData();
      form.append('file', file);
      const analyzeRes = await fetch(`${API_BASE}/analyze?_=${Date.now()}`, {
        method: 'POST',
        body: form,
        cache: 'no-store',
        headers: { 'Cache-Control': 'no-cache', Pragma: 'no-cache' },
      });
      if (!analyzeRes.ok) {
        const txt = await analyzeRes.text().catch(() => '');
        throw new Error(`analyze failed: ${analyzeRes.status} ${txt}`);
      }
      const { job_id } = await analyzeRes.json();
      jobIdRef.current = job_id;

      const controller = new AbortController();
      pollAbortRef.current = controller;

      const poll = async () => {
        while (true) {
          if (controller.signal.aborted) return;

          let s;
          try {
            s = await fetchStatus(jobIdRef.current);
          } catch {
            await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
            continue;
          }

          const p = s?.partial || {};
          if (Array.isArray(p.summary_timeline)) setSummaryTimeline(p.summary_timeline);
          if (typeof p.threshold_score === 'number') setThresholdScore(p.threshold_score);
          if (typeof p.final_model_score === 'number') setFinalModelScore(p.final_model_score);
          if (typeof p.intuitive_score === 'number') setIntuitionScore(p.intuitive_score);
          if (Array.isArray(p.intuition_segments)) setIntuitionSegments(p.intuition_segments);

          if (typeof s?.progress === 'number') setProgress(Math.max(0, Math.min(100, s.progress)));
          setLastUpdated(Date.now());  // <-- numeric

          if (s?.status === 'failed') {
            setStatus('failed');
            setErrorMsg(s?.error || 'Processing failed.');
            return;
          }
          if (s?.status === 'done') break;

          await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
        }

        const r = await fetchResult(jobIdRef.current);
        const p2 = r?.partial || {};
        if (Array.isArray(p2.summary_timeline)) setSummaryTimeline(p2.summary_timeline);
        if (typeof p2.threshold_score === 'number') setThresholdScore(p2.threshold_score);
        if (typeof p2.final_model_score === 'number') setFinalModelScore(p2.final_model_score);
        if (typeof p2.intuitive_score === 'number') setIntuitionScore(p2.intuitive_score);
        if (Array.isArray(p2.intuition_segments)) setIntuitionSegments(p2.intuition_segments);

        if (typeof r?.progress === 'number') setProgress(Math.max(0, Math.min(100, r.progress)));
        setLastUpdated(Date.now());    // <-- numeric
        setStatus('done');
      };

      poll().catch((e) => {
        console.error(e);
        setStatus('failed');
        setErrorMsg(String(e?.message || e));
      });
    } catch (e) {
      console.error(e);
      setStatus('failed');
      setErrorMsg(String(e?.message || e));
    }
  }, [file, fetchStatus, fetchResult, isMock]);

  return {
    file,
    videoUrl,
    status,
    progress,
    lastUpdated,          // number | null
    isMock,
    errorMsg,
    thresholdScore,
    finalModelScore,
    intuitionScore,
    intuitionSegments,
    summaryTimeline,
    onFileChange,
    handleAnalyze,
    hardReset,
  };
}
