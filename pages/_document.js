import { Html, Head, Main, NextScript } from 'next/document';

export default function Document() {
  return (
    <Html lang="en">
      <Head>
        {/* Cyberpunk-ish font (lightweight) */}
        <link rel="preconnect" href="https://fonts.gstatic.com" crossOrigin="" />
        <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&display=swap" rel="stylesheet" />
        <meta name="theme-color" content="#0b0d13" />
      </Head>
      <body>
        <Main />
        <NextScript />
      </body>
    </Html>
  );
}
