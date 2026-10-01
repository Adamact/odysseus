// Lifted verbatim from static/js/settings.js so extracted panels can post
// settings without importing the module they were extracted from.
import { invalidateSettings } from '../appConfig.js';

export async function postSettings(body) {
  try {
    return await fetch('/api/auth/settings', {
      method: 'POST',
      credentials: 'same-origin',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
  } finally {
    invalidateSettings();
  }
}
