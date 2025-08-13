import styles from '@/styles/Home.module.css';
import HeaderBar from '@/components/HeaderBar';
import UploaderPanel from '@/components/UploaderPanel';
import ProcessingStatus from '@/components/ProcessingStatus';
import ResultSection from '@/components/ResultSection';
import IntuitiveMeter from '@/components/IntuitiveMeter';
import AnomalyList from '@/components/AnomalyList';
import DecisionCard from '@/components/DecisionCard';
import ClipGrid from '@/components/ClipGrid';
import FrameGrid from '@/components/FrameGrid';
import FooterBar from '@/components/FooterBar';
import useVideoAnalysis from '@/components/useVideoAnalysis';

export default function Home() {
  const {
    // core state
    file, videoUrl, status, progress, lastUpdated, isMock, errorMsg,
    // results
    summary, intuitiveScore, anomalies, personPresent,
    thresholdScore, finalModelScore, decision, clips, frames,
    // actions
    onFileChange, handleAnalyze, hardReset,
  } = useVideoAnalysis();

  return (
    <main className={styles.page}>
      <div className={styles.bgGlow} aria-hidden />

      <HeaderBar title="AI vs Real Video Classifier" subtitle="Upload a video → Analyze → See detailed results" />

      <section className={styles.panelGrid}>
        {/* Left: Uploader side */}
        <div className={styles.panel}>
          <h2 className={styles.h2}>1) Upload</h2>
          <UploaderPanel
            file={file}
            videoUrl={videoUrl}
            status={status}
            onFileChange={onFileChange}
            onAnalyze={handleAnalyze}
            onReset={hardReset}
          />

          {status === 'processing' && (
            <ProcessingStatus
              progress={progress}
              lastUpdated={lastUpdated}
              isMock={isMock}
            />
          )}

          {status === 'failed' && errorMsg && (
            <div className={styles.errorBox}>
              <strong>Processing failed.</strong> {errorMsg}
            </div>
          )}
        </div>

        {/* Right: Results side */}
        <div className={styles.panel}>
          <h2 className={styles.h2}>2) Results</h2>

          <ResultSection title="Summary (Model 1)" updatedAt={lastUpdated} show={!!summary}>
            <p className={styles.summaryText}>{summary || '—'}</p>
          </ResultSection>

          <ResultSection title="Intuitive Score" updatedAt={lastUpdated} show={intuitiveScore != null}>
            <IntuitiveMeter value={intuitiveScore} />
          </ResultSection>

          <ResultSection title="Key Anomalies" updatedAt={lastUpdated} show={anomalies?.length > 0}>
            <AnomalyList anomalies={anomalies} />
          </ResultSection>

          <ResultSection title="Person Present" updatedAt={lastUpdated} show={personPresent != null}>
            {personPresent == null ? '—' : (
              <span className={`${styles.badge} ${personPresent ? styles.badgeYes : styles.badgeNo}`}>
                {personPresent ? 'Yes' : 'No'}
              </span>
            )}
          </ResultSection>

          <ResultSection title="Threshold Score" updatedAt={lastUpdated} show={thresholdScore != null}>
            {thresholdScore == null ? '—' : <code className={styles.codeBox}>{thresholdScore.toFixed(2)}</code>}
          </ResultSection>

          <ResultSection title="Final Model Score & Decision" updatedAt={lastUpdated} show={finalModelScore != null}>
            <DecisionCard thresholdScore={thresholdScore} finalModelScore={finalModelScore} decision={decision} />
          </ResultSection>

          <ResultSection title="Summary-aligned Sub-clips" updatedAt={lastUpdated} show={clips?.length > 0}>
            <ClipGrid clips={clips} />
          </ResultSection>

          <ResultSection title="Representative Frames" updatedAt={lastUpdated} show={frames?.length > 0}>
            <FrameGrid frames={frames} />
          </ResultSection>
        </div>
      </section>

      <FooterBar mock={isMock} />
    </main>
  );
}
