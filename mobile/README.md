# SignTalk AI — Mobile (Flutter)

## Setup

```bash
flutter pub get

# Generates lib/firebase_options.dart from your Firebase project — required
# before Firebase.initializeApp() in main.dart will work:
dart pub global activate flutterfire_cli
flutterfire configure
```

Then in `main.dart`, change:
```dart
await Firebase.initializeApp();
```
to:
```dart
await Firebase.initializeApp(options: DefaultFirebaseOptions.currentPlatform);
```
(and add `import 'firebase_options.dart';`) once that file exists.

Point the backend URL at your deployed API by passing a dart-define:
```bash
flutter run --dart-define=SIGNTALK_API_BASE_URL=https://your-backend-url
```

## Offline mode assets

Before testing the offline toggle, drop real trained models into
`assets/models/`:
- `movenet.tflite` — from `backend/convert_to_tflite.py --download_movenet`
- `bilstm.tflite` + `labels.json` — from `backend/train_bilstm.py`

See `assets/models/PLACEHOLDER_README.txt` for details.

## Structure

See the top-level project `README.md` for the full monorepo layout and
locked API contracts this app is built against.
