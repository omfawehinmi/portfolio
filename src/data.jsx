export const navLinks = [
  { id: 'home', href: '#home', icon: 'uil-estate', label: 'Home' },
  { id: 'about', href: '#about', icon: 'uil-user', label: 'About' },
  { id: 'skills', href: '#skills', icon: 'uil-file-alt', label: 'Skills' },
  { id: 'qualification', href: '#qualification', icon: 'uil-briefcase-alt', label: 'Education | Work' },
  { id: 'projects', href: '#projects', icon: 'uil-scenery', label: 'Projects' },
  { id: 'contact', href: '#contact', icon: 'uil-message', label: 'Contactme' },
]

export const skillGroups = [
  {
    icon: 'uil-brackets-curly',
    title: 'Languages',
    subtitle: 'More than 10 years',
    items: [
      { name: 'Python', percent: '95%', barClass: 'skills__python' },
      { name: 'SQL', percent: '95%', barClass: 'skills__sql' },
      { name: 'R', percent: '85%', barClass: 'skills__r' },
      { name: 'Powershell', percent: '75%', barClass: 'skills__powershell' },
      { name: 'C#', percent: '60%', barClass: 'skills__c' },
      { name: 'HTML', percent: '90%', barClass: 'skills__html' },
      { name: 'CSS', percent: '85%', barClass: 'skills__css' },
      { name: 'JavaScript', percent: '80%', barClass: 'skills__javascript' },
      { name: 'React', percent: '75%', barClass: 'skills__react' },
      { name: 'DAX (Power BI)', percent: '90%', barClass: 'skills__dax' },
    ],
  },
  {
    icon: 'uil-server-network',
    title: 'ETL / Pipeline Tools',
    subtitle: 'More than 8 years',
    items: [
      { name: 'Alteryx', percent: '90%', barClass: 'skills__alteryx' },
      { name: 'Apache Spark', percent: '85%', barClass: 'skills__apachespark' },
      { name: 'Azure Data Factory', percent: '85%', barClass: 'skills__azuredf' },
      { name: 'Hadoop', percent: '80%', barClass: 'skills__hadoop' },
      { name: 'Control-M', percent: '75%', barClass: 'skills__controlm' },
    ],
  },
  {
    icon: 'uil-clipboard-alt',
    title: 'BI Tools',
    subtitle: 'More than 8 years',
    items: [
      { name: 'Power BI (DAX)', percent: '95%', barClass: 'skills__powerbi' },
      { name: 'Tableau', percent: '95%', barClass: 'skills__tableau' },
      { name: 'Streamlit', percent: '90%', barClass: 'skills__streamlit' },
    ],
  },
  {
    icon: 'uil-swatchbook',
    title: 'Cloud & Database Systems',
    subtitle: 'More than 8 years',
    items: [
      { name: 'Snowflake', percent: '95%', barClass: 'skills__cortex' },
      { name: 'Amazon AWS', percent: '90%', barClass: 'skills__aws' },
      { name: 'Microsoft SQL Server', percent: '90%', barClass: 'skills__mss' },
      { name: 'Microsoft Azure', percent: '85%', barClass: 'skills__azuredf' },
      { name: 'Microsoft Fabric', percent: '80%', barClass: 'skills__msfabric' },
      { name: 'Oracle Database', percent: '70%', barClass: 'skills__oracle' },
    ],
  },
  {
    icon: 'uil-swatchbook',
    title: 'HRIS Systems',
    subtitle: 'More than 8 years',
    items: [
      { name: 'Workday', percent: '95%', barClass: 'skills__workday' },
      { name: 'SAP SuccessFactors', percent: '90%', barClass: 'skills__successfactors' },
      { name: 'Greenhouse', percent: '85%', barClass: 'skills__greenhouse' },
      { name: 'One Model', percent: '85%', barClass: 'skills__onemodel' },
      { name: 'Fieldglass', percent: '85%', barClass: 'skills__fieldglass' },
      { name: 'Eightfold', percent: '80%', barClass: 'skills__eightfold' },
      { name: 'Visier', percent: '80%', barClass: 'skills__visier' },
    ],
  },
  {
    icon: 'uil-brain',
    title: 'AI & Machine Learning',
    subtitle: 'More than 6 years',
    items: [
      { name: 'Snowflake Cortex LLM', percent: '95%', barClass: 'skills__cortex' },
      { name: 'TensorFlow / Scikit-learn', percent: '90%', barClass: 'skills__tensorflow' },
      { name: 'Deep Learning', percent: '90%', barClass: 'skills__deeplearning' },
      { name: 'GLM / GLMM / SARIMAX', percent: '85%', barClass: 'skills__glmm' },
      { name: 'Bayesian Statistics', percent: '80%', barClass: 'skills__bayesian' },
    ],
  },
  {
    icon: 'uil-database',
    title: 'SQL Dialects',
    subtitle: 'More than 10 years',
    items: [
      { name: 'T-SQL (SQL Server)', percent: '95%', barClass: 'skills__tsql' },
      { name: 'Snowflake SQL', percent: '95%', barClass: 'skills__snowflakesql' },
      { name: 'MySQL', percent: '90%', barClass: 'skills__mysql' },
      { name: 'PostgreSQL', percent: '85%', barClass: 'skills__postgresql' },
    ],
  },
]

export const education = [
  {
    side: 'left',
    title: 'Bachelors in Finance w/ Concentration in Capital Markets',
    subtitle: 'Bentley University',
    years: '2014 - 2018',
  },
  {
    side: 'right',
    title: 'Certificate in Database Architecture',
    subtitle: 'Bunker Hill Community College',
    years: '2018 - 2018',
  },
  {
    side: 'left',
    title: 'Certificate in SQL, PostgreSQL, MySQL',
    subtitle: 'Dataquest.io',
    years: '2018 - 2019',
  },
  {
    side: 'left',
    title: 'Certificate in Data Sceince(x2)',
    subtitle: 'Dataquest.io',
    years: '2018 - 2019',
  },
  {
    side: 'right',
    title: 'Certificate in Data Science',
    subtitle: 'Worcester Polytechnic Institute',
    years: '2021 - 2022',
  },
  {
    side: 'left',
    title: 'Masters in Data Science',
    subtitle: 'Worcester Polytechnic Institute',
    years: '2021 - 2023',
  },
]

export const work = [
  {
    side: 'right',
    wrapTime: true,
    title: 'Analyst',
    subtitle: 'SOF Financial Services',
    years: '2015-2017',
  },
  {
    side: 'left',
    title: 'Data Analyst',
    subtitle: 'Alder Partners LLC',
    years: '2017-2018',
  },
  {
    side: 'right',
    wrapTime: true,
    title: 'Shared Services Data Engineer & Senior Data Analyst',
    subtitle: 'Global Atlantic Financial Group',
    years: '2018-2020',
  },
  {
    side: 'left',
    title: 'Lead Python Developer',
    subtitle: 'Salesforce',
    years: '2020-2020',
  },
  {
    side: 'right',
    wrapTime: true,
    title: 'Senior Data Engineer',
    subtitle: 'Novanta Corporation',
    years: '2021 - 2025',
  },
  {
    side: 'left',
    title: 'Senior Data Engineer',
    subtitle: 'NetApp',
    years: '2026 - Present',
  },
]

export const projects = [
  {
    icon: 'uil-amazon',
    title: (
      <>
        Web
        <br />
        Application
        <br />
        Infrastructure
        <br />
        <span className="small-text">
          (AWS S3 Bucket,
          <br />
          AWS Elastic Beanstalk,
          <br />
          AWS Lambda, AWS EC2,
          <br />
          AWS Cloud9)
        </span>
      </>
    ),
    modalTitle: (
      <>
        Web Application Infrastructure
        <span className="small-text">
          {' '}
          (AWS S3 Bucket, AWS Elastic Beanstalk, AWS Lambda, AWS EC2, AWS Cloud9)
        </span>
      </>
    ),
    body: (
      <>
        <p>
          This project involves the development of a full-stack web application for Sof Financial Services using a React
          frontend and Flask backend. The application is hosted on Amazon AWS, utilizing various AWS services to ensure
          scalability, security, and efficiency. The frontend React application is housed in an S3 bucket, ensuring fast
          and reliable content delivery, while the Flask backend is managed on Elastic Beanstalk with load balancers for
          optimal traffic distribution. AWS Lambda functions are employed for serverless computing tasks, automating
          backend processes without the need for server management. AWS RDS with PostgreSQL serves as the database
          backbone, offering robust data management capabilities. Additional AWS services like Secret Manager, Docker,
          EC2, Route 53, AWS VPC, AWS ElastiCache, AWS Certificate Manager, CloudFront, and CloudWatch are integrated to
          enhance the application&apos;s security, performance, and monitoring. The development environment is
          streamlined using AWS Cloud9 and Visual Studio Code, enabling efficient code development, testing, and
          deployment. This project showcases a comprehensive cloud-based architecture designed to meet the high demands
          of a financial services platform.
        </p>
        <div style={{ display: 'flex', gap: '1rem', flexWrap: 'wrap', marginTop: '1rem' }}>
          <a
            href="https://www.soffinancialservices.com/"
            className="button button--flex"
            target="_blank"
            rel="noreferrer"
            style={{ flex: 1, justifyContent: 'center', minWidth: '180px' }}
          >
            soffinancialservices.com<i className="uil uil-external-link-alt button__icon"></i>
          </a>
          <a
            href="https://www.myorglab.com/"
            className="button button--flex"
            target="_blank"
            rel="noreferrer"
            style={{ flex: 1, justifyContent: 'center', minWidth: '180px' }}
          >
            myorglab.com<i className="uil uil-external-link-alt button__icon"></i>
          </a>
        </div>
      </>
    ),
  },
  {
    icon: 'uil-webcam',
    title: (
      <>
        Camera
        <br />
        Tracking:
        <br />
        Human Detection
        <br />
        <span className="small-text">(On-Premise: Yolov8, MicroPython, Stepper Motors)</span>
      </>
    ),
    modalTitle: 'Camera Tracking: Human Detection (On-Premise: Yolov8, MicroPython, Stepper Motors)',
    body: (
      <>
        <div style={{ maxWidth: '100%', margin: '0 0 1rem 0' }}>
          <div style={{ padding: '56.25% 0 0 0', position: 'relative' }}>
            <iframe
              src="https://player.vimeo.com/video/938258584?badge=0&autopause=0&player_id=0&app_id=58479"
              frameBorder="0"
              allow="autoplay; fullscreen; picture-in-picture; clipboard-write"
              style={{ position: 'absolute', top: 0, left: 0, width: '100%', height: '100%' }}
              title="Camera Tracking"
            />
          </div>
        </div>
        <p>
          This project showcases the integration of machine learning with physical infrastructure, utilizing a
          comprehensive system of hardware components and machine learning models. At its core, the Raspberry Pi Pico W
          acts as the central microcontroller unit, managing Python scripts, Bluetooth connectivity, and communication
          with devices like stepper motors and a webcam. The system employs two 12V stepper motors controlled by drivers
          to ensure precise movements necessary for operations such as robotic arms. A standard webcam serves as a
          visual sensor, capturing real-time data for monitoring and object detection. Software-wise, configuration
          scripts set up system parameters while main operational scripts coordinate tasks and integrate sensors and
          actuators with machine learning algorithms for enhanced decision-making. The project leverages pretrained
          YOLOv models for body and head detection, demonstrating the application of real-time analytics. BLE scripts
          support wireless communication, and motor control scripts facilitate precise operational control. Shared
          variables ensure consistent synchronization across various components.
        </p>
        <a
          href="https://github.com/omfawehinmi/portfolio/tree/main/machine_learning/camera_tracking"
          className="button button--flex"
          style={{ marginTop: '1rem' }}
        >
          See Github<i className="uil uil-github-alt button__icon"></i>
        </a>
      </>
    ),
  },
  {
    icon: 'uil-paint-tool',
    title: (
      <>
        Inpainting
        <br />
        Image Detection
        <br />
        <span className="small-text">
          (On-Premise:
          <br />
          Tensorflow, Imagenet)
        </span>
      </>
    ),
    modalTitle: (
      <>
        Inpainting Image Detection <span className="small-text">(On-Premise: Tensorflow, Imagenet)</span>
      </>
    ),
    body: (
      <>
        <p>
          This project detects image inpainting using convolutional neural networks and an ensemble model. It utilizes a
          dual dataset approach, combining the CIFAKE AI-Generated Synthetic Images dataset and the DeepFake Detection
          Challenge dataset. Training involves transfer learning, data augmentation, early stopping, and regularization
          methods. Models converge with binary cross-entropy loss and frozen layers. An ensemble model combines CNN
          predictions, optimized via hyperparameter tuning using hyperopt library. The project showcases CNNs and
          ensemble methods&apos; effectiveness in inpainted region detection, contributing to image forensics and
          manipulation detection.
        </p>
        <a
          href="https://github.com/omfawehinmi/portfolio/tree/main/machine_learning/inpainting"
          className="button button--flex"
          style={{ marginTop: '1rem' }}
        >
          See Github<i className="uil uil-github-alt button__icon"></i>
        </a>
      </>
    ),
  },
]
