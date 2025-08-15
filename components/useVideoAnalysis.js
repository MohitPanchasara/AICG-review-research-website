// components/useVideoAnalysis.js
// Hook to upload a video, call the backend, and poll for progressive results.
// Includes cache-busting + no-store for Vercel/CDN, and handles partial updates.

import { useCallback, useEffect, useRef, useState } from 'react';

const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || '';
const POLL_INTERVAL_MS = 900;

export default function useVideoAnalysis() {
  // ----- UI state -----
  const [file, setFile] = useState(null);
  const [videoUrl, setVideoUrl] = useState('');
  const [status, setStatus] = useState('idle'); // idle | ready | processing | done | failed
  const [progress, setProgress] = useState(0);
  const [lastUpdated, setLastUpdated] = useState(null);
  const [errorMsg, setErrorMsg] = useState('');

  // Partial/final results from backend
  const [summaryTimeline, setSummaryTimeline] = useState([]); // [[start,end,text], ...]
  const [thresholdScore, setThresholdScore] = useState(0.5);
  const [finalModelScore, setFinalModelScore] = useState(null);

  // internals
  const jobIdRef = useRef(null);
  const pollAbortRef = useRef(null);
  const isMock = !API_BASE; // if no backend configured, stay in mock mode

  // ------------- helpers (fetch with no-store & cache-buster) -------------
  const getJSON = useCallback(async (url) => {
    const res = await fetch(url, {
      method: 'GET',
      cache: 'no-store',
      headers: {
        'Cache-Control': 'no-cache',
        Pragma: 'no-cache',
      },
    });
    if (!res.ok) {
      const txt = await res.text().catch(() => '');
      throw new Error(`GET ${url} → ${res.status}: ${txt}`);
    }
    return res.json();
  }, []);

  const fetchStatus = useCallback(
    async (jid) => {
      const url = new URL(`${API_BASE}/status`);
      url.searchParams.set('job_id', jid);
      url.searchParams.set('_', Date.now().toString()); // cache-buster
      return getJSON(url.toString());
    },
    [getJSON]
  );

  const fetchResult = useCallback(
    async (jid) => {
      const url = new URL(`${API_BASE}/result`);
      url.searchParams.set('job_id', jid);
      url.searchParams.set('_', Date.now().toString());
      return getJSON(url.toString());
    },
    [getJSON]
  );

  // ------------- lifecycle / cleanup -------------
  useEffect(() => {
    return () => {
      // cleanup object URL & polling on unmount
      if (videoUrl) URL.revokeObjectURL(videoUrl);
      if (pollAbortRef.current) pollAbortRef.current.abort();
    };
  }, [videoUrl]);

  // ------------- public API -------------
  const onFileChange = useCallback((e) => {
    const f = e?.target?.files?.[0];
    if (!f) return;
    // Reset current state first
    if (videoUrl) URL.revokeObjectURL(videoUrl);
    setFile(f);
    setVideoUrl(URL.createObjectURL(f));
    setStatus('ready');
    setProgress(0);
    setErrorMsg('');
    setSummaryTimeline([]);
    setFinalModelScore(null);
    setThresholdScore(0.5);
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
      // cancel any previous polling
      if (pollAbortRef.current) pollAbortRef.current.abort();

      setStatus('processing');
      setProgress(5);
      setErrorMsg('');
      setLastUpdated(new Date());

      // 1) POST /analyze
      const form = new FormData();
      form.append('file', file);
      const analyzeRes = await fetch(`${API_BASE}/analyze?_=${Date.now()}`, {
        method: 'POST',
        body: form,
        cache: 'no-store',
        headers: {
          'Cache-Control': 'no-cache',
          Pragma: 'no-cache',
        },
      });
      if (!analyzeRes.ok) {
        const txt = await analyzeRes.text().catch(() => '');
        throw new Error(`analyze failed: ${analyzeRes.status} ${txt}`);
      }
      const { job_id } = await analyzeRes.json();
      jobIdRef.current = job_id;

      // 2) Poll /status until done, then fetch /result
      const controller = new AbortController();
      pollAbortRef.current = controller;

      const poll = async () => {
        // eslint-disable-next-line no-constant-condition
        while (true) {
          if (controller.signal.aborted) return;

          let s;
          try {
            s = await fetchStatus(jobIdRef.current);
          } catch (err) {
            // transient network error: keep trying
            await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
            continue;
          }

          // update partials
          const p = s?.partial || {};
          if (Array.isArray(p.summary_timeline)) setSummaryTimeline(p.summary_timeline);
          if (typeof p.threshold_score === 'number') setThresholdScore(p.threshold_score);
          if (typeof p.final_model_score === 'number') setFinalModelScore(p.final_model_score);

          if (typeof s?.progress === 'number') setProgress(Math.max(0, Math.min(100, s.progress)));
          setLastUpdated(new Date());

          if (s?.status === 'failed') {
            setStatus('failed');
            setErrorMsg(s?.error || 'Processing failed.');
            return;
          }
          if (s?.status === 'done') break;

          await new Promise((r) => setTimeout(r, POLL_INTERVAL_MS));
        }

        // fetch final result (no-cache + cache-buster)
        const r = await fetchResult(jobIdRef.current);
        const p2 = r?.partial || {};
        if (Array.isArray(p2.summary_timeline)) setSummaryTimeline(p2.summary_timeline);
        if (typeof p2.threshold_score === 'number') setThresholdScore(p2.threshold_score);
        if (typeof p2.final_model_score === 'number') setFinalModelScore(p2.final_model_score);

        if (typeof r?.progress === 'number') setProgress(Math.max(0, Math.min(100, r.progress)));
        setLastUpdated(new Date());
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
    // state
    file,
    videoUrl,
    status,
    progress,
    lastUpdated,
    isMock,
    errorMsg,
    thresholdScore,
    finalModelScore,
    summaryTimeline,

    // handlers
    onFileChange,
    handleAnalyze,
    hardReset,
  };
}
